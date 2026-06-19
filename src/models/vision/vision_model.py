"""
SuPreM vision model for donor-liver macrosteatosis (fat >= 30%) from CT.

Backbone : SwinUNETR encoder (MONAI), initialised from SuPreM AbdomenAtlas
           pretraining, fine-tuned with an ordinal head on 96^3 liver volumes.
Input    : a single-channel 96x96x96 volume, intensities in [0, 1].

The frozen checkpoint (weights/vision_suprem.pt) holds the *fine-tuned* encoder
and head, so prediction needs only this file -- the external SuPreM backbone is
required for training only.

Usage
  # regenerate the checkpoint (needs GPU, the 96^3 image npz, and the SuPreM
  # backbone .pth from https://github.com/MrGiovanni/SuPreM)
  python vision_model.py train --images images_96.npz --csv donors.csv \
         --suprem supervised_suprem_swinunetr_2100.pth --out weights/vision_suprem.pt

  # score your own volumes (npz with keys: uuids[str], images[N,96,96,96] in [0,1])
  python vision_model.py predict --images your_volumes.npz --out predictions.csv
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from monai.networks.nets import SwinUNETR

TARGET        = "MACRO_FAT_LI_DON"
FAT_THRESHOLD = 30
NUM_CLASSES   = 5
FEATURE_SIZE  = 48          # SwinUNETR base
N_PRETRAIN    = 25          # AbdomenAtlas organ count (output head of the backbone)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = DEVICE.startswith("cuda")


# ------------------------------------------------------------------
# Labels / ordinal helpers
# ------------------------------------------------------------------
def bin_5class(arr):
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, -1, dtype=np.int64)
    out[a == 0] = 0
    out[(a >= 1) & (a <= 9)] = 1
    out[(a >= 10) & (a <= 29)] = 2
    out[(a >= 30) & (a <= 49)] = 3
    out[a >= 50] = 4
    return out


def ordinal_targets(y, k):
    rng = torch.arange(k - 1, device=y.device)
    return (y.unsqueeze(1) > rng.unsqueeze(0)).float()


def ordinal_to_pos_prob(logits):
    """Ordinal threshold logits -> P(class >= 3) = P(fat >= 30%)."""
    thresh = torch.sigmoid(logits)           # (B, K-1), thresh[:,k] = P(y > k)
    return thresh[:, 2]                       # P(y > 2) = P(y >= 3)


def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def _load_suprem(swin, path):
    """Load SuPreM SwinUNETR weights, auto-detecting the key prefix."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "net", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    target = set(swin.state_dict().keys())
    prefix = ""
    for k in sd:
        if "swinViT." in k:
            prefix = k[:k.find("swinViT.")]; break
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    missing, unexpected = swin.load_state_dict(sd, strict=False)
    loaded = sum(1 for k in target if k.startswith("swinViT.") and k not in missing)
    total  = sum(1 for k in target if k.startswith("swinViT."))
    print(f"  SuPreM backbone: loaded {loaded}/{total} swinViT keys")


class BinarySuPreM(nn.Module):
    """SwinUNETR encoder + ordinal head -> (B, K-1) threshold logits."""
    def __init__(self, suprem_path=None, dropout=0.3):
        super().__init__()
        self.swin = SwinUNETR(in_channels=1, out_channels=N_PRETRAIN,
                              feature_size=FEATURE_SIZE, spatial_dims=3,
                              use_checkpoint=False)
        if suprem_path:
            _load_suprem(self.swin, suprem_path)
        feat_dim = 16 * FEATURE_SIZE
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES - 1),
        )

    def forward(self, x):
        # autocast lives inside forward so AMP also works under nn.DataParallel:
        # DataParallel runs forward in worker threads where a main-thread
        # autocast context would not take effect.
        with torch.amp.autocast(device_type="cuda", enabled=IS_CUDA):
            h = self.swin.swinViT(x, self.swin.normalize)[-1]   # (B, feat, d, h, w)
            return self.head(h.flatten(2).mean(dim=2))


class VolDataset(Dataset):
    def __init__(self, imgs, idx, y, train=False):
        self.imgs, self.idx, self.y, self.train = imgs, np.asarray(idx), np.asarray(y), train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        v = torch.from_numpy(np.ascontiguousarray(
            self.imgs[self.idx[i]].astype(np.float32))).unsqueeze(0)
        if self.train:
            if torch.rand(1).item() < 0.5:            # superior/inferior flip only
                v = torch.flip(v, dims=[2])
            v = v + torch.randn_like(v) * 0.01
        return v, torch.tensor(int(self.y[i]))


@torch.no_grad()
def infer(model, loader):
    model.eval()
    ys, ps = [], []
    for v, y in loader:
        logit = model(v.to(DEVICE))           # autocast handled inside forward
        ps.append(ordinal_to_pos_prob(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ------------------------------------------------------------------
# Train (single deployable run on all labelled data) / Predict
# ------------------------------------------------------------------
def load_aligned(images, csv):
    npz = np.load(images, allow_pickle=True)
    uu, imgs = npz["uuids"].astype(str), npz["images"]
    assert imgs.shape[1:] == (96, 96, 96), f"expected 96^3, got {imgs.shape[1:]}"
    df = pd.read_csv(csv); df["uuid"] = df["uuid"].astype(str)
    idx = {u: i for i, u in enumerate(uu)}
    df = df[df["uuid"].isin(idx)].reset_index(drop=True)
    rows = np.array([idx[u] for u in df["uuid"]])
    return df, imgs[rows]


def fit_vision(imgs, tr, va, y5, suprem_path, epochs=100, patience=25, batch_size=8):
    """Fine-tune SuPreM SwinUNETR on tr, early-stop on va. Returns the best
    model and the Youden threshold from va. Shared by vision_model and full_model
    so the full ensemble's image branch trains on the same split as the meta-learner.

    Uses all visible GPUs via nn.DataParallel (each batch is split across them),
    so a batch that does not fit on one card fits when spread over several."""
    tl = DataLoader(VolDataset(imgs, tr, y5[tr], train=True), batch_size=batch_size, shuffle=True)
    vl = DataLoader(VolDataset(imgs, va, y5[va]), batch_size=batch_size)

    model = BinarySuPreM(suprem_path=suprem_path).to(DEVICE)
    enc = {id(p) for p in model.swin.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if id(p) in enc], "lr": 5e-5},
         {"params": [p for p in model.parameters() if id(p) not in enc], "lr": 1e-4}],
        weight_decay=1e-4)

    # Wrap for multi-GPU AFTER building the optimizer (so the param groups still
    # reference the real module). The optimizer updates that same module in place,
    # so model.state_dict() stays clean (no "module." prefix) for saving / predict.
    n_gpu = torch.cuda.device_count() if IS_CUDA else 0
    net = nn.DataParallel(model) if n_gpu > 1 else model
    if n_gpu > 1:
        print(f"  DataParallel across {n_gpu} GPUs  "
              f"(batch {batch_size} -> ~{batch_size // n_gpu}/GPU)")

    sched = SequentialLR(opt, [LinearLR(opt, 0.1, total_iters=5),
                               CosineAnnealingLR(opt, T_max=max(1, epochs - 5), eta_min=1e-6)],
                         milestones=[5])
    # per-threshold pos_weight (capped at 5)
    pw = torch.tensor([min(max((y5[tr] <= k).sum(), 1) / max((y5[tr] > k).sum(), 1), 5.0)
                       for k in range(NUM_CLASSES - 1)], dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=IS_CUDA)

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        for v, y in tl:
            v, y = v.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(v).float(), ordinal_targets(y, NUM_CLASSES))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        vy, vp = infer(net, vl)
        auc = roc_auc_score((vy >= 3).astype(int), vp)
        if auc > best_auc + 1e-6:
            best_auc, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        print(f"  epoch {ep:03d}  val_auc={auc:.4f}  (best={best_auc:.4f})")
        if wait >= patience:
            print(f"  early stop @ {ep}"); break

    model.load_state_dict(best_state)
    vy, vp = infer(net, vl)
    return model, youden_threshold((vy >= 3).astype(int), vp), float(best_auc)


def train(args):
    df, imgs = load_aligned(args.images, args.csv)
    raw = df[TARGET].values.astype(float)
    keep = ~np.isnan(raw)
    imgs, raw = imgs[keep], raw[keep]
    y5 = bin_5class(raw); y2 = (raw >= FAT_THRESHOLD).astype(int)
    tr, va = train_test_split(np.arange(len(y5)), test_size=0.1,
                              random_state=42, stratify=y5)
    print(f"Train {len(tr)} / val {len(va)}  (pos>={FAT_THRESHOLD}%: {int(y2.sum())})")

    model, thr, best_auc = fit_vision(imgs, tr, va, y5, args.suprem,
                                      args.epochs, args.patience, args.batch_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"kind": "vision_suprem", "state_dict": model.state_dict(),
                "threshold": thr, "feature_size": FEATURE_SIZE,
                "fat_threshold_pct": FAT_THRESHOLD, "best_val_auc": best_auc},
               args.out)
    print(f"Saved {args.out}  (val AUC={best_auc:.4f}, threshold={thr:.3f})")


def predict(args):
    ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    npz = np.load(args.images, allow_pickle=True)
    uu, imgs = npz["uuids"].astype(str), npz["images"]
    model = BinarySuPreM(suprem_path=None).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    loader = DataLoader(VolDataset(imgs, np.arange(len(imgs)), np.zeros(len(imgs))),
                        batch_size=8)
    _, p = infer(model, loader)
    out = pd.DataFrame({"uuid": uu, "p_macrosteatosis": p,
                        "high_risk": (p >= ckpt["threshold"]).astype(int)})
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} volumes -> {args.out}  "
          f"(high-risk: {int(out.high_risk.sum())}, threshold={ckpt['threshold']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--images", required=True, help="npz: uuids[str], images[N,96,96,96] in [0,1]")
    t.add_argument("--csv", required=True, help="table with uuid + MACRO_FAT_LI_DON")
    t.add_argument("--suprem", required=True, help="SuPreM SwinUNETR backbone .pth")
    t.add_argument("--out", default="weights/vision_suprem.pt")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=32, dest="batch_size",
                   help="split across all visible GPUs via DataParallel")
    t.set_defaults(func=train)
    p = sub.add_parser("predict")
    p.add_argument("--weights", default="weights/vision_suprem.pt")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(func=predict)
    args = ap.parse_args()
    args.func(args)
