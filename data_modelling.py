import json
import logging
import os
logger = logging.getLogger(__name__)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from settings import HYPERPARAMETER_GRID, N_SPLITS, RANDOM_STATE, FAIRNESS_MARGIN
from constants import BEST_HYPERPARAMETERS_PATH, FAIRNESS_REPORT_PATH
import warnings

def model_pipeline(X, y, preprocessor):
    logger.info("Starting model training and hyperparameter search...")
    models = {
        "logistic": LogisticRegression(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }

    best_score = 0
    best_model = None
    best_params = None

    for name, model in models.items():
        logger.info(f"Training model: {name}")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid={f"classifier__{k}": v for k, v in HYPERPARAMETER_GRID[name].items()},
            scoring=make_scorer(f1_score, pos_label=1),
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        )
        logger.info(f"Running GridSearchCV for {name}...")
        grid.fit(X, y)
        logger.info(f"Completed GridSearchCV for {name}, Best F1: {grid.best_score_:.4f}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_params = grid.best_params_

    # Train fairness-aware model
    logger.info("Training fairness-aware model (Fairlearn ExponentiatedGradient)...")
    sensitive_features = X["sex"].astype(str) + "_" + X["race"].astype(str)
    X_fair = X.drop(columns=["sex", "race"])

    # Ensure all categorical features in X_fair are encoded for fairness model
    X_fair = X_fair.copy()
    for col in X_fair.select_dtypes(include="object").columns:
        X_fair[col] = X_fair[col].astype("category").cat.codes

    fair_model = ExponentiatedGradient(
        estimator=LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
        constraints=DemographicParity()
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="fairlearn")
        fair_model.fit(X_fair, y, sensitive_features=sensitive_features)
    y_pred_fair = fair_model.predict(X_fair)
    fair_f1 = f1_score(y, y_pred_fair, pos_label=1)
    fair_precision = precision_score(y, y_pred_fair, pos_label=1)
    fair_recall = recall_score(y, y_pred_fair, pos_label=1)
    logger.info(f"Fairness-aware model F1: {fair_f1:.4f}, Precision: {fair_precision:.4f}, Recall: {fair_recall:.4f}")

    with open(FAIRNESS_REPORT_PATH, "w") as f:
        f.write("Fairness-Aware Model (ExponentiatedGradient with DemographicParity)\n")
        f.write(f"F1 Score: {fair_f1:.4f}\n")
        f.write(f"Precision: {fair_precision:.4f}\n")
        f.write(f"Recall: {fair_recall:.4f}\n")

    if fair_f1 > FAIRNESS_MARGIN * best_score:
        logger.info("Fairness-aware model selected due to comparable performance and improved fairness.")
        best_score = fair_f1
        best_model = fair_model
        best_params = {
            "model": "fairlearn_exponentiated_gradient",
            "f1_score": fair_f1,
            "precision": fair_precision,
            "recall": fair_recall
        }

    # Save best model and params
    model_name = type(best_model.named_steps['classifier']).__name__
    logger.info(f"Best model: {model_name}, F1 Score: {best_score:.4f}")
    logger.info("Saving best model and hyperparameters...")

    with open(BEST_HYPERPARAMETERS_PATH, "w") as f:
        json.dump(best_params, f)

    logger.info("Model training pipeline completed.")
    return best_model
