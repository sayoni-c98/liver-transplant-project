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

# ==================================================
# OOF PLATT CALIBRATION
# ==================================================

def get_oof_platt(X_train, y5_train, ybin_train, seed, fixed_params, seed_dir):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_raw = np.zeros(len(X_train), dtype=float)
    rows = []

    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(X_train, y5_train), start=1):
        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]

        y5_tr = y5_train.iloc[tr_idx]
        y5_val = y5_train.iloc[val_idx]

        pre = make_preprocessor(FULL_MODEL_FEATURES)
        X_tr = pre.fit_transform(X_tr_raw)
        X_val = pre.transform(X_val_raw)

        sw = compute_sample_weight(class_weight="balanced", y=y5_tr)

        model = make_xgb(seed + 20000 + fold_id, fixed_params)
        model = fit_xgb_with_early_stopping(
            model,
            X_tr,
            y5_tr,
            X_val,
            y5_val,
            sample_weight=sw,
        )

        prob_val_raw, _ = get_binary_prob_from_5class(model, X_val)
        oof_raw[val_idx] = prob_val_raw

        for j, global_idx in enumerate(val_idx):
            rows.append({
                "seed": seed,
                "fold": fold_id,
                "row_id_train": int(global_idx),
                "y_binary_true": int(ybin_train.iloc[global_idx]),
                "prob_raw_oof": float(prob_val_raw[j]),
            })

    platt = fit_platt_scaler(oof_raw, ybin_train.values)
    oof_platt = apply_platt_scaler(platt, oof_raw)

    sweep_platt = threshold_sweep(ybin_train.values, oof_platt)
    best_thr = get_best_mcc_threshold(sweep_platt)

    cv_oof_pred_df = pd.DataFrame(rows)
    cv_oof_pred_df["prob_platt_oof"] = oof_platt
    cv_oof_pred_df.to_csv(
        seed_dir / "cv_oof_predictions_raw_and_platt.csv",
        index=False,
    )

    sweep_platt["seed"] = seed
    sweep_platt["split"] = "cv_oof_platt"
    sweep_platt.to_csv(
        seed_dir / "cv_oof_threshold_sweep_platt.csv",
        index=False,
    )

    return platt, best_thr, cv_oof_pred_df, sweep_platt


# ==================================================
# FINAL MODEL TRAINING
# ==================================================

def train_final_model(X_train, y5_train, seed, fixed_params):
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.20,
        random_state=seed,
        stratify=y5_train,
    )

    X_tr_raw = X_train.iloc[tr_idx]
    X_val_raw = X_train.iloc[val_idx]

    y_tr = y5_train.iloc[tr_idx]
    y_val = y5_train.iloc[val_idx]

    pre = make_preprocessor(FULL_MODEL_FEATURES)

    X_tr = pre.fit_transform(X_tr_raw)
    X_val = pre.transform(X_val_raw)

    sw = compute_sample_weight(class_weight="balanced", y=y_tr)

    model = make_xgb(seed, fixed_params)
    model = fit_xgb_with_early_stopping(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        sample_weight=sw,
    )

    return pre, model

# ==================================================
# ONE-SEED FINAL EVALUATION
# ==================================================

def run_one_final_seed(df, fixed_params, seed, output_dir):
    seed_dir = output_dir / "stage2_final_10seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["y_5class"],
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    X_train = train_df[FULL_MODEL_FEATURES].copy()
    X_test = test_df[FULL_MODEL_FEATURES].copy()

    y5_train = train_df["y_5class"].astype(int)
    y5_test = test_df["y_5class"].astype(int)

    ybin_train = train_df["y_binary"].astype(int)
    ybin_test = test_df["y_binary"].astype(int)

    print(
        f"{MODEL_NAME} | seed {seed} | "
        f"train={len(train_df)} test={len(test_df)} | "
        f"test_pos={ybin_test.sum()} test_neg={(1 - ybin_test).sum()}",
        flush=True,
    )

    platt, best_thr, cv_oof_pred_df, cv_sweep_platt = get_oof_platt(
        X_train=X_train,
        y5_train=y5_train,
        ybin_train=ybin_train,
        seed=seed,
        fixed_params=fixed_params,
        seed_dir=seed_dir,
    )

    pre_final, model_final = train_final_model(
        X_train=X_train,
        y5_train=y5_train,
        seed=seed,
        fixed_params=fixed_params,
    )

    joblib.dump(model_final, seed_dir / "final_model.pkl")
    joblib.dump(pre_final, seed_dir / "final_preprocessor.pkl")
    joblib.dump(platt, seed_dir / "platt_scaler.pkl")

    X_test_trans = pre_final.transform(X_test)
    prob_test_raw, p5_test = get_binary_prob_from_5class(model_final, X_test_trans)
    prob_test_platt = apply_platt_scaler(platt, prob_test_raw)

    test_metrics = compute_metrics(ybin_test.values, prob_test_platt, best_thr)

    pre_brier = float(brier_score_loss(ybin_test.values, prob_test_raw))
    pre_ece = expected_calibration_error(ybin_test.values, prob_test_raw)
    pre_slope, pre_intercept = calibration_slope_intercept(
        ybin_test.values,
        prob_test_raw,
    )

    test_sweep_platt = threshold_sweep(ybin_test.values, prob_test_platt)
    test_sweep_platt["seed"] = seed
    test_sweep_platt["split"] = "test_platt"
    test_sweep_platt["cv_selected_threshold_by_mcc"] = best_thr
    test_sweep_platt.to_csv(
        seed_dir / "test_threshold_sweep_platt.csv",
        index=False,
    )

    pred_df = pd.DataFrame({
        "model": MODEL_NAME,
        "seed": seed,
        "biopsy_macro_fat": test_df[TARGET_COL].values,
        "y_5class_true": y5_test.values,
        "y_binary_true": ybin_test.values,
        "prob_binary_raw": prob_test_raw,
        "prob_binary_platt": prob_test_platt,
        "pred_binary_platt_cv_mcc_threshold": (
            prob_test_platt >= best_thr
        ).astype(int),
        "cv_selected_threshold_by_mcc": best_thr,
        "p_class_0_clean": p5_test[:, 0],
        "p_class_1_1to9": p5_test[:, 1],
        "p_class_2_10to29": p5_test[:, 2],
        "p_class_3_30to49": p5_test[:, 3],
        "p_class_4_50plus": p5_test[:, 4],
    })

    if UUID_COL in test_df.columns:
        pred_df.insert(2, UUID_COL, test_df[UUID_COL].values)

    if DONOR_ID_COL in test_df.columns:
        insert_position = 3 if UUID_COL in pred_df.columns else 2
        pred_df.insert(insert_position, DONOR_ID_COL, test_df[DONOR_ID_COL].values)

    pred_df.to_csv(seed_dir / "test_predictions.csv", index=False)

    detail = {
        "model": MODEL_NAME,
        "seed": seed,
        "calibration_method": "platt",
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_pos_binary": int(ybin_train.sum()),
        "train_neg_binary": int((1 - ybin_train).sum()),
        "test_pos_binary": int(ybin_test.sum()),
        "test_neg_binary": int((1 - ybin_test).sum()),
        "n_features_raw": len(FULL_MODEL_FEATURES),
        "cv_selected_threshold_by_mcc": best_thr,
        "pre_brier": pre_brier,
        "pre_ece": pre_ece,
        "pre_cal_slope": pre_slope,
        "pre_cal_intercept": pre_intercept,
        **test_metrics,
    }

    for k, v in fixed_params.items():
        detail[f"fixed_param_{k}"] = v

    pd.DataFrame([detail]).to_csv(
        seed_dir / "final_test_metrics.csv",
        index=False,
    )

    print(
        f"{MODEL_NAME} | seed {seed} | "
        f"AUC={test_metrics['roc_auc']:.4f} | "
        f"PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"MCC={test_metrics['mcc']:.4f} | "
        f"Brier(pre)={pre_brier:.4f} "
        f"Brier(platt)={test_metrics['brier']:.4f}",
        flush=True,
    )

    return {
        "detail": detail,
        "pred": pred_df,
        "cv_sweep": cv_sweep_platt,
        "test_sweep": test_sweep_platt,
    }

# ==================================================
# AGGREGATION
# ==================================================

def aggregate_outputs(output_dir, seed_outputs, fixed_params):
    agg_dir = output_dir / "aggregated_results"
    agg_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame([x["detail"] for x in seed_outputs])
    pred_df = pd.concat([x["pred"] for x in seed_outputs], ignore_index=True)
    cv_sweep_df = pd.concat([x["cv_sweep"] for x in seed_outputs], ignore_index=True)
    test_sweep_df = pd.concat([x["test_sweep"] for x in seed_outputs], ignore_index=True)

    detail_df.to_csv(agg_dir / "final_test_metrics_all_seeds.csv", index=False)
    pred_df.to_csv(agg_dir / "test_predictions_all_seeds.csv", index=False)
    cv_sweep_df.to_csv(agg_dir / "cv_oof_threshold_sweep_all_seeds.csv", index=False)
    test_sweep_df.to_csv(agg_dir / "test_threshold_sweep_all_seeds.csv", index=False)

    metric_cols = [
        "roc_auc", "pr_auc",
        "brier", "ece", "cal_slope", "cal_intercept",
        "pre_brier", "pre_ece",
        "accuracy", "f1",
        "sensitivity", "specificity", "ppv", "npv", "mcc",
        "sens_at_90spec",
        "cv_selected_threshold_by_mcc",
    ]

    summary_rows = []

    for metric in metric_cols:
        if metric not in detail_df.columns:
            continue

        mean = detail_df[metric].mean()
        std = detail_df[metric].std()

        summary_rows.append({
            "model": MODEL_NAME,
            "metric": metric,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "formatted": fmt_mean_std(mean, std),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(agg_dir / "final_summary_mean_std_4decimals.csv", index=False)

    paper_row = {"model": MODEL_NAME}

    for metric in metric_cols:
        if metric in detail_df.columns:
            paper_row[metric] = fmt_mean_std(
                detail_df[metric].mean(),
                detail_df[metric].std(),
            )

    for k, v in fixed_params.items():
        paper_row[f"fixed_param_{k}"] = v

    pd.DataFrame([paper_row]).to_csv(
        agg_dir / "paper_ready_summary_wide.csv",
        index=False,
    )

    print("=" * 90)
    print("AGGREGATED RESULTS")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    return paper_row


# ==================================================
# MAIN
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train full multimodal XGBoost model with Platt calibration."
    )

    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="Path to input CSV containing required features and target column.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/xgboost_full_model",
        help="Directory where outputs will be saved.",
    )

    parser.add_argument(
        "--random_search_iter",
        type=int,
        default=RANDOM_SEARCH_ITER,
        help="Number of random-search hyperparameter candidates.",
    )

    parser.add_argument(
        "--tune_jobs",
        type=int,
        default=TUNE_PARALLEL_JOBS,
        help="Parallel jobs for hyperparameter tuning.",
    )

    parser.add_argument(
        "--eval_jobs",
        type=int,
        default=EVAL_PARALLEL_JOBS,
        help="Parallel jobs for final seed evaluation.",
    )

    parser.add_argument(
        "--xgb_jobs",
        type=int,
        default=XGB_N_JOBS,
        help="Threads per XGBoost model.",
    )

    return parser.parse_args()


def main():
    global RANDOM_SEARCH_ITER
    global TUNE_PARALLEL_JOBS
    global EVAL_PARALLEL_JOBS
    global XGB_N_JOBS

    args = parse_args()

    RANDOM_SEARCH_ITER = args.random_search_iter
    TUNE_PARALLEL_JOBS = args.tune_jobs
    EVAL_PARALLEL_JOBS = args.eval_jobs
    XGB_N_JOBS = args.xgb_jobs

    data_csv = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FULL MULTIMODAL XGBOOST PIPELINE")
    print("=" * 90)
    print(f"Model:       {MODEL_NAME}")
    print(f"Data CSV:    {data_csv}")
    print(f"Output dir:  {output_dir}")
    print(f"Features:    {len(FULL_MODEL_FEATURES)}")
    print(f"Calibration: Platt scaling on OOF training probabilities")
    print("=" * 90)

    df = load_data(data_csv)

    pd.DataFrame({
        "rank": range(1, len(FULL_MODEL_FEATURES) + 1),
        "feature": FULL_MODEL_FEATURES,
    }).to_csv(output_dir / "feature_list_used.csv", index=False)

    fixed_params = choose_fixed_params(df, output_dir)

    print("=" * 90)
    print("STAGE 2 — FINAL 10-SEED EVALUATION")
    print("=" * 90)

    seed_outputs = Parallel(
        n_jobs=EVAL_PARALLEL_JOBS,
        backend="loky",
        verbose=5,
    )(
        delayed(run_one_final_seed)(
            df,
            fixed_params,
            seed,
            output_dir,
        )
        for seed in SEEDS
    )

    aggregate_outputs(output_dir, seed_outputs, fixed_params)

    print("\nDONE.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
