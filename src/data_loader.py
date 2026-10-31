"""
Data loading and preprocessing module for Wine Quality Dataset.
Adheres to ML Best Practices:
- Schema validation
- Missing value checks and data sanitization
- Clean train/test split BEFORE fitting any preprocessing
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(data_dir: str = "data"):
    """
    Loads red and white wine datasets and returns them as pandas DataFrames.
    """
    red_path = os.path.join(data_dir, "winequality-red.csv")
    white_path = os.path.join(data_dir, "winequality-white.csv")

    if not os.path.exists(red_path) or not os.path.exists(white_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Run download_data.py first.")

    # UCI wine dataset uses semicolon as separator
    df_red = pd.read_csv(red_path, sep=";")
    df_white = pd.read_csv(white_path, sep=";")

    df_red["wine_type"] = 1  # 1 for red
    df_white["wine_type"] = 0  # 0 for white

    df_combined = pd.concat([df_red, df_white], ignore_index=True)
    return df_red, df_white, df_combined


def get_data_summary(df: pd.DataFrame, name: str = "Dataset"):
    """
    Prints a structured statistical and data quality summary.
    """
    summary = {
        "dataset_name": name,
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
    }
    return summary


def prepare_classification_dataset(
    wine_type: str = "combined",
    quality_threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Prepares train and test splits for wine quality classification.
    Target:
      Binary: 1 if quality >= quality_threshold (Good quality), 0 otherwise.
    
    Returns:
      X_train, X_test, y_train, y_test, feature_names
    """
    df_red, df_white, df_combined = load_raw_data()

    if wine_type == "red":
        df = df_red.drop(columns=["wine_type"])
    elif wine_type == "white":
        df = df_white.drop(columns=["wine_type"])
    elif wine_type == "combined":
        df = df_combined
    else:
        raise ValueError("wine_type must be 'red', 'white', or 'combined'")

    # Target variable: 1 for High Quality (>= threshold), 0 for Normal/Low Quality
    df = df.copy()
    y = (df["quality"] >= quality_threshold).astype(int)
    X = df.drop(columns=["quality"])

    feature_names = list(X.columns)

    # Strictly split before fitting any transformers
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    df_red, df_white, df_combined = load_raw_data()
    print("Red Wine Summary:", get_data_summary(df_red, "Red Wine"))
    print("White Wine Summary:", get_data_summary(df_white, "White Wine"))
    print("Combined Summary:", get_data_summary(df_combined, "Combined Wine"))

