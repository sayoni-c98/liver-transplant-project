"""
Full multimodal ensemble for donor-liver macrosteatosis (fat >= 30%).

  base learners : LightGBM + XGBoost + TabPFN (tabular)  +  SuPreM SwinUNETR (CT)
  meta-learner  : logistic regression over the four binary >=30% probabilities

This is the no-vision tabular ensemble with the SuPreM image probability added as
a fourth base learner. Inference needs both the tabular feature row and the 96^3
CT volume for each donor (joined by uuid).

The frozen artifact is two files:
  weights/full.pkl          tabular base learners + 4-input meta-learner + threshold
  weights/full_vision.pt    SuPreM image branch (fine-tuned on the same split)

The image branch is trained on the meta-learner's training split, so the stacked
probabilities the meta-learner sees are out-of-sample (no leakage).

Usage
  # regenerate (needs GPU, the 96^3 image npz, and the SuPreM backbone .pth)
  python full_model.py train --csv donors.csv --hu hu_histograms.csv \
         --images images_96.npz --suprem supervised_suprem_swinunetr_2100.pth

  # score your own donors (CSV of features + npz of volumes, joined on uuid)
  python full_model.py predict --csv your_donors.csv --images your_volumes.npz \
         --out predictions.csv
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

import common as C
# vision_model lives in the sibling ../vision folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))
import vision_model as V


def load_joined(csv, hu, images):
    """Return df, image array, and 96^3 volumes aligned row-for-row on uuid."""
    df = pd.read_csv(csv); df["uuid"] = df["uuid"].astype(str)
    if hu and os.path.isfile(hu):
        h = pd.read_csv(hu); h["uuid"] = h["uuid"].astype(str)
        df = df.merge(h, on="uuid", how="inner")
    npz = np.load(images, allow_pickle=True)
    uu, imgs = npz["uuids"].astype(str), npz["images"]
    assert imgs.shape[1:] == (96, 96, 96), f"expected 96^3, got {imgs.shape[1:]}"
    idx = {u: i for i, u in enumerate(uu)}
    df = df[df["uuid"].isin(idx)].reset_index(drop=True)
    rows = np.array([idx[u] for u in df["uuid"]])
    return df, imgs[rows]


def vision_probs(model, imgs, threshold=None):
    loader = DataLoader(V.VolDataset(imgs, np.arange(len(imgs)), np.zeros(len(imgs))),
                        batch_size=8)
    _, p = V.infer(model, loader)
    return p


def train(args):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    df, imgs = load_joined(args.csv, args.hu, args.images)
    cols = C.feature_columns(df)
    raw = df[C.TARGET].values.astype(float)
    keep = ~np.isnan(raw)
    df, imgs, raw = df[keep].reset_index(drop=True), imgs[keep], raw[keep]
    X  = C.to_numeric_matrix(df, cols)
    y5 = C.bin_5class(raw)
    y2 = (raw >= C.FAT_THRESHOLD).astype(int)
    print(f"Joined: {X.shape[0]} donors x {len(cols)} features + 96^3 CT  "
          f"(pos>={C.FAT_THRESHOLD}%: {int(y2.sum())})")

    # 80/20: base learners on the 80% fit set, meta-learner on the 20% hold-out.
    fit_i, hold_i = train_test_split(np.arange(len(y5)), test_size=0.2,
                                     random_state=42, stratify=y5)
    # Carve a vision early-stop val out of the fit set.
    vtr, vva = train_test_split(fit_i, test_size=0.1, random_state=42, stratify=y5[fit_i])

    print("Fitting LightGBM ..."); lgbm = C.fit_lgb(X[fit_i], y5[fit_i], 42)
    print("Fitting XGBoost ...");  xgbm = C.fit_xgb(X[fit_i], y5[fit_i], 42)
    print(f"Fitting TabPFN ({args.device}) ...")
    tab = C.fit_tabpfn(X[fit_i], y5[fit_i], 42, args.device)
    print("Fine-tuning SuPreM vision branch ...")
    vmodel, _, v_auc = V.fit_vision(imgs, vtr, vva, y5, args.suprem,
                                    args.epochs, args.patience, args.batch_size)
    print(f"  vision val AUC={v_auc:.4f}")

    base_hold = np.column_stack([
        C.binary_from_5class(lgbm, X[hold_i]),
        C.binary_from_5class(xgbm, X[hold_i]),
        C.binary_from_5class(tab,  X[hold_i]),
        vision_probs(vmodel, imgs[hold_i]),
    ])
    meta = C.fit_meta_lr(base_hold, y2[hold_i])
    p_hold = C.meta_predict(meta, base_hold)
    thr = C.mcc_optimal_threshold(y2[hold_i], p_hold)
    print(f"Hold-out  ROC AUC={roc_auc_score(y2[hold_i], p_hold):.4f}  "
          f"PR AUC={average_precision_score(y2[hold_i], p_hold):.4f}  thr={thr:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vision_out = os.path.splitext(args.out)[0].replace("full", "full_vision") + ".pt"
    if vision_out == args.out:
        vision_out = "weights/full_vision.pt"
    torch.save({"kind": "vision_suprem", "state_dict": vmodel.state_dict(),
                "feature_size": V.FEATURE_SIZE}, vision_out)
    with open(args.out, "wb") as f:
        pickle.dump({"kind": "full", "feature_cols": cols,
                     "lgb": lgbm, "xgb": xgbm, "tabpfn": tab,
                     "meta_lr": meta, "threshold": thr,
                     "vision_checkpoint": os.path.basename(vision_out),
                     "fat_threshold_pct": C.FAT_THRESHOLD}, f)
    print(f"Saved {args.out} and {vision_out}")


def predict(args):
    with open(args.weights, "rb") as f:
        b = pickle.load(f)
    df, imgs = load_joined(args.csv, args.hu, args.images)
    missing = [c for c in b["feature_cols"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing {len(missing)} feature columns, e.g. {missing[:8]}")
    X = C.to_numeric_matrix(df, b["feature_cols"])

    vpath = os.path.join(os.path.dirname(args.weights), b["vision_checkpoint"])
    ckpt = torch.load(vpath, map_location=V.DEVICE, weights_only=False)
    vmodel = V.BinarySuPreM(suprem_path=None).to(V.DEVICE)
    vmodel.load_state_dict(ckpt["state_dict"])

    base = np.column_stack([
        C.binary_from_5class(b["lgb"], X),
        C.binary_from_5class(b["xgb"], X),
        C.binary_from_5class(b["tabpfn"], X),
        vision_probs(vmodel, imgs),
    ])
    p = C.meta_predict(b["meta_lr"], base)
    out = pd.DataFrame({"uuid": df["uuid"].astype(str),
                        "p_macrosteatosis": p,
                        "high_risk": (p >= b["threshold"]).astype(int)})
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} donors -> {args.out}  "
          f"(high-risk: {int(out.high_risk.sum())}, threshold={b['threshold']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--csv", required=True, help="table with uuid + MACRO_FAT_LI_DON + the 218 features")
    t.add_argument("--hu",  default=None, help="optional HU-histogram table to merge on uuid")
    t.add_argument("--images", required=True, help="npz: uuids[str], images[N,96,96,96] in [0,1]")
    t.add_argument("--suprem", required=True, help="SuPreM SwinUNETR backbone .pth")
    t.add_argument("--out", default="weights/full.pkl")
    t.add_argument("--device", default="cpu", help="TabPFN device")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                   help="vision branch batch, split across all visible GPUs")
    t.set_defaults(func=train)
    p = sub.add_parser("predict")
    p.add_argument("--weights", default="weights/full.pkl")
    p.add_argument("--csv", required=True)
    p.add_argument("--hu", default=None, help="optional HU-histogram table to merge on uuid")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(func=predict)
    args = ap.parse_args()
    args.func(args)
