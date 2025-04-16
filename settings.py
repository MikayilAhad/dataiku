# General
RANDOM_STATE = 42
TARGET_COLUMN = "income"

# Class labels
POSITIVE_CLASS = " 50000+."
NEGATIVE_CLASS = " - 50000."

# Fairness-related
SENSITIVE_FEATURES = ["sex", "race"]

# Cross-validation
N_SPLITS = 5

# SMOTE
SMOTE_K_NEIGHBORS = 5

# FAIRNESS THRESHOLD
FAIRNESS_MARGIN = 0.95

# Hyperparameter grids
HYPERPARAMETER_GRID = {
    "logistic": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["liblinear"]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6, 10]
    }
}
