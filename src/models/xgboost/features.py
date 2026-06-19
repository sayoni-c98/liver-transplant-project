"""
Feature definitions for the multimodal XGBoost model.

Model:
    Clinical features (25)
    + Original median HU (1)
    + Topological features (150)

Total = 176 raw features
"""

# ==================================================
# TOPOLOGICAL FEATURES
# ==================================================

TOPO_FEATURES = [f"b{i}" for i in range(150)]


# ==================================================
# HU FEATURE
# ==================================================

HU_FEATURE = "original_dataset_median_hu"


# ==================================================
# CLINICAL FEATURES
# ==================================================

CLINICAL_FEATURES = [
    # Continuous variables
    "AGE_DON",
    "BUN_DON",
    "SGOT_DON",
    "SGPT_DON",
    "liver_volume",
    "sat_volume_mm3",
    "skeletal_muscle_volume_mm3",
    "vat_volume_mm3",
    "BMI_DON_CALC",
    "CREAT_DON",
    "KDPI",
    "imat_volume_mm3",
    "muscle_quality_ratio_imat_to_sm",
    "vat_sat_ratio",
    "TBILI_DON",

    # Categorical variables
    "COD_CAD_DON",
    "APRIcat",
    "CPR_ADMIN",
    "FIB4cat",
    "HIST_COCAINE_DON",
    "HIST_HYPERTENS_DON",
    "NAFLDcat",
    "donor_race",
    "BARDcat",
    "HIST_CIG_DON",
]


# ==================================================
# CATEGORICAL FEATURES
# ==================================================

CATEGORICAL_COLS = [
    "COD_CAD_DON",
    "APRIcat",
    "CPR_ADMIN",
    "FIB4cat",
    "HIST_COCAINE_DON",
    "HIST_HYPERTENS_DON",
    "NAFLDcat",
    "donor_race",
    "BARDcat",
    "HIST_CIG_DON",
]


# ==================================================
# FULL MULTIMODAL FEATURE SET
# ==================================================

FULL_MODEL_FEATURES = (
    CLINICAL_FEATURES +
    [HU_FEATURE] +
    TOPO_FEATURES
)


# ==================================================
# FEATURE COUNTS
# ==================================================

N_CLINICAL_FEATURES = len(CLINICAL_FEATURES)
N_TOPO_FEATURES = len(TOPO_FEATURES)
N_TOTAL_FEATURES = len(FULL_MODEL_FEATURES)
