"""
No-vision tabular ensemble for donor-liver macrosteatosis (fat >= 30%).

  base learners : LightGBM + XGBoost + TabPFN  (trained on 5-class fat bins)
  meta-learner  : logistic regression over the three binary >=30% probabilities

Features (218, all numeric): clinical / donor-history, body-composition &
organ-HU statistics, HU histogram, and 150 topological persistence features
(f0..f149). See README for the required column list.

Usage
  # regenerate the frozen weights (authors only; needs the training table)
  python no_vision_model.py train  --out weights/no_vision.pkl

  # score your own donors (one row per donor, same feature columns)
  python no_vision_model.py predict --csv your_donors.csv --out predictions.csv

The frozen model is a single pickle holding the fitted base learners, the
meta-learner, the feature order and the operating threshold. TabPFN is a tabular
foundation model: the fitted object carries its training table, so no separate
training data is needed at predict time.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

import common as C

DEF_OUT = "weights/no_vision.pkl"
SEED    = 42


def load_table(csv, hu):
    df = pd.read_csv(csv); df["uuid"] = df["uuid"].astype(str)
    if hu and os.path.isfile(hu):
        h = pd.read_csv(hu); h["uuid"] = h["uuid"].astype(str)
        df = df.merge(h, on="uuid", how="inner")
    return df


def train(args):
    df = load_table(args.csv, args.hu)
    cols = C.feature_columns(df)
    raw = df[C.TARGET].values.astype(float)
    keep = ~np.isnan(raw)
    df, raw = df[keep].reset_index(drop=True), raw[keep]
    X  = C.to_numeric_matrix(df, cols)
    y5 = C.bin_5class(raw)
    y2 = (raw >= C.FAT_THRESHOLD).astype(int)
    print(f"Train table: {X.shape[0]} donors x {len(cols)} features  "
          f"(pos>={C.FAT_THRESHOLD}%: {int(y2.sum())})")

    # 5-fold out-of-fold stacking: the meta-learner and threshold are fit on
    # out-of-fold base predictions (leakage-free), then the base learners are
    # refit on all donors for the strongest deployable model.
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    oof = np.zeros((len(y5), 3), dtype=np.float64)
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y5), 1):
        lk = C.fit_lgb(X[tr], y5[tr], SEED)
        xk = C.fit_xgb(X[tr], y5[tr], SEED)
        tk = C.fit_tabpfn(X[tr], y5[tr], SEED, args.device)
        for j, m in enumerate((lk, xk, tk)):
            oof[va, j] = C.binary_from_5class(m, X[va])
        print(f"  fold {k}/5  ROC AUC: lgb={roc_auc_score(y2[va], oof[va,0]):.4f} "
              f"xgb={roc_auc_score(y2[va], oof[va,1]):.4f} tab={roc_auc_score(y2[va], oof[va,2]):.4f}")

    meta = C.fit_meta_lr(oof, y2)
    p_oof = C.meta_predict(meta, oof)
    thr = C.mcc_optimal_threshold(y2, p_oof)
    print(f"Out-of-fold  ROC AUC={roc_auc_score(y2, p_oof):.4f}  "
          f"PR AUC={average_precision_score(y2, p_oof):.4f}  thr={thr:.3f}")

    print("Refitting base learners on all donors ...")
    lgbm = C.fit_lgb(X, y5, SEED)
    xgbm = C.fit_xgb(X, y5, SEED)
    tab  = C.fit_tabpfn(X, y5, SEED, args.device)

    bundle = {"kind": "no_vision", "feature_cols": cols,
              "lgb": lgbm, "xgb": xgbm, "tabpfn": tab,
              "meta_lr": meta, "threshold": thr,
              "fat_threshold_pct": C.FAT_THRESHOLD}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {args.out}")


def predict(args):
    with open(args.weights, "rb") as f:
        b = pickle.load(f)
    df = pd.read_csv(args.csv)
    uuid = df["uuid"].astype(str) if "uuid" in df.columns else pd.Series(np.arange(len(df)))
    missing = [c for c in b["feature_cols"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing {len(missing)} required feature columns, "
                         f"e.g. {missing[:8]}")
    X = C.to_numeric_matrix(df, b["feature_cols"])
    base = np.column_stack([C.binary_from_5class(m, X)
                            for m in (b["lgb"], b["xgb"], b["tabpfn"])])
    p = C.meta_predict(b["meta_lr"], base)
    out = pd.DataFrame({"uuid": uuid,
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
    t.add_argument("--out", default=DEF_OUT)
    t.add_argument("--device", default="cpu", help="TabPFN device (cpu keeps the artifact portable)")
    t.set_defaults(func=train)
    p = sub.add_parser("predict")
    p.add_argument("--weights", default=DEF_OUT)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(func=predict)
    args = ap.parse_args()
    args.func(args)
