"""
Platt calibration utilities.

Platt scaling is fit on out-of-fold training predictions only, then applied to test-set probabilities.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_scaler(oof_raw, y_true):
    """
    Fit Platt scaling using logistic regression on OOF raw probabilities.
    """
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(np.asarray(oof_raw).reshape(-1, 1), y_true)
    return platt


def apply_platt_scaler(platt, prob_raw):
    """
    Apply a fitted Platt scaler to raw probabilities.
    """
    return platt.predict_proba(np.asarray(prob_raw).reshape(-1, 1))[:, 1]
