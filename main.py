import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from data_preparation import load_and_prepare_data
from data_modelling import model_pipeline
from model_assessment import assess_model
from settings import TARGET_COLUMN


def main():
    # Load and preprocess
    train_df, test_df, preprocessor = load_and_prepare_data()
    logger.info("Data loaded and split into train and test sets.")
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # Train and tune models (pipeline includes preprocessing and SMOTE)
    best_model = model_pipeline(X_train, y_train, preprocessor)
    logger.info("Best model trained and selected.")
    print(f"The best model is: {type(best_model).__name__}")

    # Extract feature names after preprocessing
    try:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        logger.warning("Preprocessor does not support get_feature_names_out. Inferring feature names from transformed data.")
        transformed_X_train = best_model.named_steps['preprocessor'].transform(X_train)
        feature_names = [f"feature_{i}" for i in range(transformed_X_train.shape[1])]

    # Evaluate and save results
    assess_model(best_model, X_test, y_test, feature_names)
    logger.info("Model evaluated on test set and results saved.")

if __name__ == "__main__":
    main()
