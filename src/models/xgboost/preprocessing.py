"""
Preprocessing utilities for the XGBoost model.

This module builds the sklearn preprocessing pipeline used before XGBoost:
    - numeric features are passed through unchanged
    - categorical features are one-hot encoded
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL_COLS


def make_preprocessor(feature_list):
    """
    Create a preprocessing transformer for the selected feature list.

    Numeric features:
        Passed through unchanged.

    Categorical features:
        One-hot encoded with unknown categories ignored at test time.
    """
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_list]
    num_cols = [c for c in feature_list if c not in cat_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
