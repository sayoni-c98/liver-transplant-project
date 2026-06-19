#!/usr/bin/env python3
"""
Full multimodal XGBoost model for liver steatosis classification.

Model:
    25 clinical features
    + original median HU
    + 150 topology features

Target:
    Binary classification of macrovesicular steatosis >= 30%.

Calibration:
    Platt scaling fitted on out-of-fold training predictions only.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from xgboost import XGBClassifier

from features import FULL_MODEL_FEATURES
from preprocessing import make_preprocessor
from calibration import fit_platt_scaler, apply_platt_scaler
from metrics import (
    compute_metrics,
    threshold_sweep,
    get_best_mcc_threshold,
    expected_calibration_error,
    calibration_slope_intercept,
    fmt_mean_std,
)

warnings.filterwarnings("ignore")


# ==================================================
# COLUMN NAMES
# ==================================================

TARGET_COL = "MACRO_FAT_LI_DON"
UUID_COL = "uuid"
DONOR_ID_COL = "donor_id"


# ==================================================
# EXPERIMENT SETTINGS
# ==================================================

MODEL_NAME = "Clinical + HU + Topology"

SEEDS = [42, 52, 62, 72, 77, 87, 97, 107, 117, 127]

TEST_SIZE = 0.20
N_SPLITS = 5

RANDOM_SEARCH_ITER = 100
PARAM_SAMPLER_SEED = 2026
EARLY_STOPPING_ROUNDS = 50

XGB_N_JOBS = 4
TUNE_PARALLEL_JOBS = 4
EVAL_PARALLEL_JOBS = 4


# ==================================================
# HYPERPARAMETER SEARCH SPACE
# ==================================================

PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 750, 1000, 1200, 1500, 2000],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 5, 8],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0],
}


# ==================================================
# LABEL FUNCTIONS
# ==================================================

def macro_fat_to_5class(x):
    if pd.isna(x):
        return np.nan

    x = float(x)

    if x == 0:
        return 0
    if 1 <= x <= 9:
        return 1
    if 10 <= x <= 29:
        return 2
    if 30 <= x <= 49:
        return 3
    if x >= 50:
        return 4

    return np.nan


def macro_fat_to_binary(x):
    if pd.isna(x):
        return np.nan

    return int(float(x) >= 30)


# ==================================================
# MODEL HELPERS
# ==================================================

def clean_params(params):
    params = dict(params)

    for k in ["n_estimators", "max_depth", "min_child_weight"]:
        params[k] = int(params[k])

    for k in [
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]:
        params[k] = float(params[k])

    return params


def make_xgb(seed, params):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=XGB_N_JOBS,
        **clean_params(params),
    )


def fit_xgb_with_early_stopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weight=None,
):
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
    except TypeError:
        model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    return model


def get_binary_prob_from_5class(model, X):
    """
    Binary probability:
        P(steatosis >= 30%) = P(class 3) + P(class 4)
    """
    p5 = model.predict_proba(X)
    prob_binary = p5[:, 3] + p5[:, 4]

    return prob_binary, p5

# ==================================================
# DATA LOADING
# ==================================================

def load_data(data_csv):
    df = pd.read_csv(data_csv)

    required = [TARGET_COL] + FULL_MODEL_FEATURES
    optional_id_cols = [UUID_COL, DONOR_ID_COL]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["y_5class"] = df[TARGET_COL].apply(macro_fat_to_5class)
    df["y_binary"] = df[TARGET_COL].apply(macro_fat_to_binary)

    before = len(df)
    df = df.dropna(subset=[TARGET_COL, "y_5class", "y_binary"] + FULL_MODEL_FEATURES).copy()
    after = len(df)

    df["y_5class"] = df["y_5class"].astype(int)
    df["y_binary"] = df["y_binary"].astype(int)

    print("=" * 90)
    print("DATA LOADED")
    print("=" * 90)
    print(f"Input CSV:          {data_csv}")
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna:  {after}")
    print(f"Features:           {len(FULL_MODEL_FEATURES)}")
    print("\n5-class label distribution:")
    print(df["y_5class"].value_counts().sort_index())
    print("\nBinary label distribution:")
    print(df["y_binary"].value_counts().sort_index())
    print("=" * 90)

    return df


# ==================================================
# RANDOM SEARCH
# ==================================================

def evaluate_one_param_one_seed(df, param_id, params, seed):
    train_idx, _ = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["y_5class"],
    )

    train_df = df.loc[train_idx].reset_index(drop=True)

    X_train = train_df[FULL_MODEL_FEATURES].copy()
    y5_train = train_df["y_5class"].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []

    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(X_train, y5_train), start=1):
        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]

        y_tr = y5_train.iloc[tr_idx]
        y_val = y5_train.iloc[val_idx]
        y_val_bin = (y_val >= 3).astype(int)

        pre = make_preprocessor(FULL_MODEL_FEATURES)
        X_tr = pre.fit_transform(X_tr_raw)
        X_val = pre.transform(X_val_raw)

        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        model = make_xgb(seed + 100000 + param_id * 100 + fold_id, params)
        model = fit_xgb_with_early_stopping(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            sample_weight=sw,
        )

        prob_val, _ = get_binary_prob_from_5class(model, X_val)

        fold_rows.append({
            "param_id": param_id,
            "seed": seed,
            "fold": fold_id,
            "fold_roc_auc": float(roc_auc_score(y_val_bin, prob_val)),
            "fold_pr_auc": float(average_precision_score(y_val_bin, prob_val)),
            **clean_params(params),
        })

    return fold_rows


def choose_fixed_params(df, output_dir):
    tune_dir = output_dir / "stage1_random_search"
    tune_dir.mkdir(parents=True, exist_ok=True)

    param_list = list(ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=RANDOM_SEARCH_ITER,
        random_state=PARAM_SAMPLER_SEED,
    ))

    pd.DataFrame([
        {"param_id": i + 1, **clean_params(p)}
        for i, p in enumerate(param_list)
    ]).to_csv(tune_dir / "random_search_parameter_list.csv", index=False)

    print("=" * 90)
    print("STAGE 1 — RANDOM SEARCH")
    print("=" * 90)
    print(f"Candidates: {RANDOM_SEARCH_ITER}")
    print(f"Seeds:      {len(SEEDS)}")
    print(f"Folds:      {N_SPLITS}")
    print(f"Total fits: {RANDOM_SEARCH_ITER * len(SEEDS) * N_SPLITS}")
    print("=" * 90)

    all_rows_nested = Parallel(
        n_jobs=TUNE_PARALLEL_JOBS,
        backend="loky",
        verbose=5,
    )(
        delayed(evaluate_one_param_one_seed)(
            df,
            param_id=i + 1,
            params=params,
            seed=seed,
        )
        for i, params in enumerate(param_list)
        for seed in SEEDS
    )

    rows = [r for block in all_rows_nested for r in block]
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(tune_dir / "random_search_fold_level_results.csv", index=False)

    group_cols = ["param_id"] + list(PARAM_DISTRIBUTIONS.keys())

    summary_df = (
        fold_df
        .groupby(group_cols, as_index=False)
        .agg(
            mean_cv_pr_auc=("fold_pr_auc", "mean"),
            std_cv_pr_auc=("fold_pr_auc", "std"),
            mean_cv_roc_auc=("fold_roc_auc", "mean"),
            std_cv_roc_auc=("fold_roc_auc", "std"),
            n_evals=("fold", "count"),
        )
        .sort_values(
            ["mean_cv_pr_auc", "mean_cv_roc_auc"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    summary_df.to_csv(tune_dir / "random_search_summary_by_param.csv", index=False)

    best_row = summary_df.iloc[0]
    best_params = clean_params({k: best_row[k] for k in PARAM_DISTRIBUTIONS.keys()})

    with open(tune_dir / "fixed_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nFixed best parameters:")
    print(json.dumps(best_params, indent=4))

    return best_params

