"""
Shared pieces for the donor-liver macrosteatosis models.

Three released models share this file:
  - no_vision_model.py  : tabular stacked ensemble (LGB + XGB + TabPFN -> LR)
  - vision_model.py     : SuPreM SwinUNETR image model (standalone)
  - full_model.py       : tabular ensemble + SuPreM image probability -> LR

Target: donor liver macroscopic steatosis, MACRO_FAT_LI_DON >= 30% (biopsy).
Base learners are trained on the 5-class fat bin and collapsed to the binary
>=30% probability; a logistic-regression meta-learner stacks them.

Hyper-parameters here are identical to the published benchmark so regenerated
weights reproduce the paper models.
"""

import warnings
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from tabpfn import TabPFNClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef

# LightGBM 4.x records placeholder feature names ("Column_0"...) even when fit on
# a numpy array, so predicting on a numpy array raises a cosmetic sklearn warning.
# Column order is enforced by name before conversion, so predictions are correct.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


TARGET        = "MACRO_FAT_LI_DON"
ID_COLS       = {"uuid", "donor_id"}
FAT_THRESHOLD = 30
NUM_CLASSES   = 5


# ------------------------------------------------------------------
# Labels and feature selection
# ------------------------------------------------------------------
def bin_5class(arr):
    """{0}->0, [1,9]->1, [10,29]->2, [30,49]->3, [50,inf)->4. NaN -> -1."""
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, -1, dtype=np.int64)
    out[a == 0]                = 0
    out[(a >= 1)  & (a <= 9)]  = 1
    out[(a >= 10) & (a <= 29)] = 2
    out[(a >= 30) & (a <= 49)] = 3
    out[a >= 50]               = 4
    return out


def feature_columns(df):
    """Every column except ids and the target. All features are numeric."""
    return [c for c in df.columns if c not in (ID_COLS | {TARGET})]


def to_numeric_matrix(df, cols):
    """Select feature columns in fixed order as float32. NaNs are preserved;
    LightGBM / XGBoost / TabPFN all handle missing values natively."""
    return df[cols].astype(np.float32).values


def binary_from_5class(model, X):
    """P(class in {3,4}) = P(fat >= 30%), robust to absent classes."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pos = np.zeros(len(X), dtype=np.float64)
    for c in (3, 4):
        if c in classes:
            pos += proba[:, classes.index(c)]
    return pos


# ------------------------------------------------------------------
# Base learners (identical config to the benchmark)
# ------------------------------------------------------------------
def fit_lgb(X, y5, seed):
    Xtr, Xva, ytr, yva = train_test_split(X, y5, test_size=0.1,
                                          random_state=seed, stratify=y5)
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=NUM_CLASSES, metric="multi_logloss",
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def fit_xgb(X, y5, seed):
    Xtr, Xva, ytr, yva = train_test_split(X, y5, test_size=0.1,
                                          random_state=seed, stratify=y5)
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=ytr)
    m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=NUM_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, early_stopping_rounds=50, verbosity=0,
    )
    m.fit(Xtr, ytr, sample_weight=cw[ytr], eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_tabpfn(X, y5, seed, device):
    m = TabPFNClassifier(device=device, random_state=seed,
                         ignore_pretraining_limits=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X.astype(np.float32), y5)
    return m


def fit_meta_lr(base_val, y2_val):
    """LR over base-learner binary probabilities (columns). Falls back to the
    mean if a degenerate (single-class) validation fold is passed."""
    if y2_val.sum() == 0 or (y2_val == 0).sum() == 0:
        return None
    return LogisticRegression(C=1.0, max_iter=2000).fit(base_val, y2_val)


def meta_predict(meta_lr, base_X):
    if meta_lr is None:
        return base_X.mean(axis=1)
    return meta_lr.predict_proba(base_X)[:, 1]


def mcc_optimal_threshold(y, p):
    """Threshold on p that maximises MCC against y (chosen on held-out probs)."""
    if len(np.unique(y)) < 2:
        return 0.5
    best_mcc, best_thr = -2.0, 0.5
    for thr in np.unique(p):
        yp = (p >= thr).astype(int)
        if yp.sum() == 0 or yp.sum() == len(yp):
            continue
        m = matthews_corrcoef(y, yp)
        if m > best_mcc:
            best_mcc, best_thr = m, thr
    return float(best_thr)
