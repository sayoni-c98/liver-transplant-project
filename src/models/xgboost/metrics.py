"""
Metric utilities for binary steatosis classification.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)


THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def sensitivity_at_fixed_specificity(y_true, y_prob, target_spec=0.90):
    best_sens = 0.0
    best_thr = 1.0

    for thr in sorted(THRESHOLDS, reverse=True):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if spec >= target_spec:
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best_sens = float(sens)
            best_thr = float(thr)
            break

    return best_sens, best_thr


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])

        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / len(y_true))


def calibration_slope_intercept(y_true, y_prob):
    prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    log_odds = np.log(prob_clipped / (1 - prob_clipped))

    try:
        cal = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        cal.fit(log_odds.reshape(-1, 1), y_true)
        slope = float(cal.coef_[0][0])
        intercept = float(cal.intercept_[0])
    except Exception:
        slope, intercept = np.nan, np.nan

    return slope, intercept


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    ece = expected_calibration_error(y_true, y_prob)
    cal_slope, cal_intercept = calibration_slope_intercept(y_true, y_prob)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": ece,
        "cal_slope": cal_slope,
        "cal_intercept": cal_intercept,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        
    }


def threshold_sweep(y_true, y_prob):
    return pd.DataFrame([
        compute_metrics(y_true, y_prob, float(thr))
        for thr in THRESHOLDS
    ])


def get_best_mcc_threshold(sweep_df):
    best = sweep_df.sort_values(
        ["mcc", "pr_auc", "roc_auc", "brier"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return float(best["threshold"])


def fmt_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"
