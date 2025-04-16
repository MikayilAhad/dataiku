import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from constants import *
from settings import TARGET_COLUMN

def assess_model(best_model, X_test, y_test, feature_names):
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save metrics
    with open(MODEL_REPORT_PATH, "w") as f:
        f.write(f"Model: {type(best_model.named_steps['classifier']).__name__}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")

    # Save model
    joblib.dump(best_model, FINAL_MODEL_PATH)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)

    # Save feature importance
    if hasattr(best_model.named_steps["classifier"], "feature_importances_"):
        importances = best_model.named_steps["classifier"].feature_importances_
        if len(importances) != len(feature_names):
            feature_names = feature_names[:len(importances)]

        # Map transformed feature indices back to original feature names if possible
        if hasattr(best_model.named_steps['preprocessor'], 'transformers_'):
            original_feature_names = []
            for name, transformer, columns in best_model.named_steps['preprocessor'].transformers_:
                if transformer != 'drop' and hasattr(transformer, 'get_feature_names_out'):
                    original_feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    original_feature_names.extend(columns)
            feature_names = original_feature_names

        # Order features by importance in descending order and keep only the top 10
        sorted_indices = (-importances).argsort()[:10]
        feature_names = [feature_names[i] for i in sorted_indices]
        importances = importances[sorted_indices]

        # Reverse the order for barh to display the most important feature on top
        feature_names = feature_names[::-1]
        importances = importances[::-1]

        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, importances)
        plt.title("Top 10 Feature Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH)
