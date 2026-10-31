"""
Wine Quality Model Training and Benchmark Module.
Implements ML Best Practices:
- Data leakage prevention via Pipeline and StratifiedKFold
- Baseline models (Dummy, Logistic Regression)
- Advanced tree ensembles (Random Forest, Gradient Boosting, Extra Trees)
- Comprehensive metric tracking (Accuracy, Precision, Recall, F1, ROC-AUC)
- Model persistence via joblib
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import prepare_classification_dataset


def get_model_candidates():
    """
    Defines model pipelines to evaluate.
    Scaling is included in the pipeline to prevent data leakage during CV.
    """
    models = {
        "Dummy (Majority Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistic Regression (Linear Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
        ]),
        "Extra Trees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42))
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=150, max_depth=8, random_state=42))
        ])
    }
    return models


def evaluate_models_cross_validation(models, X_train, y_train, cv_splits=5):
    """
    Runs Stratified K-Fold Cross Validation across all model candidates.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []
    print(f"--- Running {cv_splits}-Fold Stratified Cross Validation ---")
    for name, pipeline in models.items():
        t0 = time.time()
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0

        res = {
            "Model": name,
            "CV Accuracy Mean": np.mean(scores["test_accuracy"]),
            "CV Accuracy Std": np.std(scores["test_accuracy"]),
            "CV Precision Mean": np.mean(scores["test_precision"]),
            "CV Recall Mean": np.mean(scores["test_recall"]),
            "CV F1 Mean": np.mean(scores["test_f1"]),
            "CV ROC-AUC Mean": np.mean(scores["test_roc_auc"]),
            "Fit Time (s)": elapsed
        }
        results.append(res)
        print(f"[{name}] F1: {res['CV F1 Mean']:.4f} | ROC-AUC: {res['CV ROC-AUC Mean']:.4f} | Time: {elapsed:.2f}s")

    df_results = pd.DataFrame(results).sort_values(by="CV F1 Mean", ascending=False)
    return df_results


def train_and_evaluate_test_set(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Fits the selected best pipeline on the full training set and evaluates on unseen test data.
    """
    print("\n--- Training Best Model on Full Training Set ---")
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline, "predict_proba"):
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = best_pipeline.decision_function(X_test)

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Normal/Low (0)", "High Quality (1)"])

    return test_metrics, cm, cr, y_pred, y_proba


def run_pipeline(wine_type="combined", output_dir="models"):
    """
    Full pipeline run: data prep, model comparison, best model selection, test evaluation, and saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = prepare_classification_dataset(wine_type=wine_type)
    print(f"Dataset '{wine_type}': Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}, Features = {len(feature_names)}")

    models = get_model_candidates()
    cv_summary = evaluate_models_cross_validation(models, X_train, y_train)
    print("\n--- Cross-Validation Leaderboard ---")
    print(cv_summary.to_string(index=False))

    cv_summary.to_csv("reports/cv_model_comparison.csv", index=False)

    # Best model by F1 Score (handling class imbalance)
    best_model_name = cv_summary.iloc[0]["Model"]
    print(f"\nTop performing model: {best_model_name}")
    best_pipeline = models[best_model_name]

    test_metrics, cm, cr, y_pred, y_proba = train_and_evaluate_test_set(
        best_pipeline, X_train, y_train, X_test, y_test
    )

    print("\n--- Test Set Evaluation ---")
    for metric_name, val in test_metrics.items():
        print(f"{metric_name}: {val:.4f}")

    print("\nClassification Report:\n", cr)
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    model_save_path = os.path.join(output_dir, f"best_wine_model_{wine_type}.joblib")
    artifact_payload = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "wine_type": wine_type,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary.to_dict(orient="records"),
    }
    joblib.dump(artifact_payload, model_save_path)
    print(f"\nModel and metadata saved successfully to {model_save_path}")

    # Save test predictions for evaluation and visualization
    test_eval_df = X_test.copy()
    test_eval_df["y_true"] = y_test
    test_eval_df["y_pred"] = y_pred
    test_eval_df["y_proba"] = y_proba
    test_eval_df.to_csv("reports/test_predictions.csv", index=False)

    return artifact_payload, cv_summary, test_metrics, cm


if __name__ == "__main__":
    run_pipeline("combined")
