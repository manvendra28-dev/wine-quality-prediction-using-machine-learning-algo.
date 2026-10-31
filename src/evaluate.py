"""
Evaluation and Visualization Module for Wine Quality ML.
Generates comprehensive visual reports:
- Model comparison chart
- Confusion matrix
- ROC curve
- Feature importance analysis (via Permutation Importance)
- Feature correlation heatmap
"""

import os
import sys

# Configure writable cache for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.abspath(".tmp_mpl")
os.makedirs(".tmp_mpl", exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_raw_data, prepare_classification_dataset


def generate_visual_reports(
    model_path: str = "models/best_wine_model_combined.joblib",
    reports_dir: str = "reports/figures"
):
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    print(f"Loading artifact from {model_path}...")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    model_name = artifact["model_name"]
    feature_names = artifact["feature_names"]

    # 1. Plot CV Model Comparison
    cv_csv_path = "reports/cv_model_comparison.csv"
    if os.path.exists(cv_csv_path):
        df_cv = pd.read_csv(cv_csv_path)
        plt.figure(figsize=(10, 6))
        plot_df = pd.melt(
            df_cv,
            id_vars=["Model"],
            value_vars=["CV Accuracy Mean", "CV Precision Mean", "CV Recall Mean", "CV F1 Mean", "CV ROC-AUC Mean"],
            var_name="Metric",
            value_name="Score"
        )
        plot_df["Metric"] = plot_df["Metric"].str.replace("CV ", "").str.replace(" Mean", "")
        
        ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
        plt.title("Cross-Validation Performance Comparison Across Algorithms", fontsize=14, weight="bold", pad=15)
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        cv_plot_path = os.path.join(reports_dir, "model_comparison.png")
        plt.savefig(cv_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {cv_plot_path}")

    # 2. Confusion Matrix & ROC Curve from Test Predictions
    preds_path = "reports/test_predictions.csv"
    if os.path.exists(preds_path):
        df_preds = pd.read_csv(preds_path)
        y_true = df_preds["y_true"]
        y_pred = df_preds["y_pred"]
        y_proba = df_preds["y_proba"]

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot_matrix = np.array([
            [f"{count}\n({pct:.1%})" for count, pct in zip(c_row, p_row)]
            for c_row, p_row in zip(cm, cm_percent)
        ])
        sns.heatmap(
            cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=True,
            xticklabels=["Normal/Low (0)", "High Quality (1)"],
            yticklabels=["Normal/Low (0)", "High Quality (1)"]
        )
        plt.title(f"Confusion Matrix: {model_name}", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("True Class", fontsize=12)
        plt.tight_layout()
        cm_plot_path = os.path.join(reports_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_plot_path}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_score = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"{model_name} (AUC = {roc_score:.3f})")
        plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, weight="bold", pad=15)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        roc_plot_path = os.path.join(reports_dir, "roc_curve.png")
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_plot_path}")

    # 3. Permutation Feature Importance
    print("Computing Permutation Feature Importance...")
    X_train, X_test, y_train, y_test, _ = prepare_classification_dataset(wine_type=artifact["wine_type"])
    perm_importance = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
    )
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    importance_df = pd.DataFrame({
        "Feature": np.array(feature_names)[sorted_idx],
        "Importance": perm_importance.importances_mean[sorted_idx],
        "Std": perm_importance.importances_std[sorted_idx]
    })
    importance_df.to_csv("reports/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df, x="Importance", y="Feature",
        palette="viridis", errorbar=None
    )
    plt.title(f"Top Predictive Physicochemical Features ({model_name})", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Permutation Feature Importance (Decrease in Accuracy)", fontsize=12)
    plt.ylabel("Wine Attribute", fontsize=12)
    plt.tight_layout()
    fi_plot_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fi_plot_path}")

    # 4. Correlation Heatmap
    _, _, df_combined = load_raw_data()
    plt.figure(figsize=(11, 9))
    corr = df_combined.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-0.8, vmax=0.8, square=True, linewidths=0.5
    )
    plt.title("Wine Physicochemical Properties Correlation Matrix", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    corr_plot_path = os.path.join(reports_dir, "correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_plot_path}")

    print("\nAll evaluation figures generated successfully.")


if __name__ == "__main__":
    generate_visual_reports()
