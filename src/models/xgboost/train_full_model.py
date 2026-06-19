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
