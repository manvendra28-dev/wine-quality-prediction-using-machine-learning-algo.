"""
Inference and Prediction Module for Wine Quality ML Model.
Allows single-sample and batch prediction from physicochemical attributes.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class WineQualityPredictor:
    def __init__(self, model_path: str = "models/best_wine_model_combined.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first using train.py.")
        
        self.artifact = joblib.load(model_path)
        self.pipeline = self.artifact["pipeline"]
        self.model_name = self.artifact["model_name"]
        self.feature_names = self.artifact["feature_names"]
        self.wine_type = self.artifact["wine_type"]

    def predict_sample(self, sample_dict: dict) -> dict:
        """
        Predicts quality for a single sample dictionary.
        Returns prediction (0/1), class label, and probability.
        """
        # Validate that all required features are present
        missing_features = [f for f in self.feature_names if f not in sample_dict]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Construct DataFrame in exact feature order
        df_sample = pd.DataFrame([sample_dict])[self.feature_names]

        pred = self.pipeline.predict(df_sample)[0]
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(df_sample)[0][1]
        else:
            proba = float(self.pipeline.decision_function(df_sample)[0])

        label = "High Quality (Score >= 7)" if pred == 1 else "Normal / Low Quality (Score < 7)"
        return {
            "prediction": int(pred),
            "label": label,
            "high_quality_probability": float(np.round(proba, 4)),
            "model_used": self.model_name
        }


def run_demo():
    print("=" * 60)
    print("      WINE QUALITY CHECKER - MACHINE LEARNING DEMO       ")
    print("=" * 60)

    predictor = WineQualityPredictor()
    print(f"Loaded Trained Model: {predictor.model_name}")

    # Sample 1: Premium high-alcohol, low volatile acidity white wine
    premium_wine = {
        "fixed acidity": 6.8,
        "volatile acidity": 0.18,
        "citric acid": 0.36,
        "residual sugar": 1.4,
        "chlorides": 0.034,
        "free sulfur dioxide": 38.0,
        "total sulfur dioxide": 105.0,
        "density": 0.9898,
        "pH": 3.25,
        "sulphates": 0.52,
        "alcohol": 12.8,
        "wine_type": 0  # White
    }

    # Sample 2: Poor low-alcohol, high volatile acidity (vinegary) red wine
    poor_wine = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.88,
        "citric acid": 0.00,
        "residual sugar": 2.6,
        "chlorides": 0.098,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 42.0,
        "density": 0.9976,
        "pH": 3.51,
        "sulphates": 0.44,
        "alcohol": 9.4,
        "wine_type": 1  # Red
    }

    print("\n[Sample 1: High-tier White Wine]")
    for k, v in premium_wine.items():
        print(f"  {k:22s}: {v}")
    res1 = predictor.predict_sample(premium_wine)
    print(f"  --> Prediction: {res1['label']}")
    print(f"  --> Confidence: {res1['high_quality_probability'] * 100:.1f}%\n")

    print("[Sample 2: Low-tier Red Wine]")
    for k, v in poor_wine.items():
        print(f"  {k:22s}: {v}")
    res2 = predictor.predict_sample(poor_wine)
    print(f"  --> Prediction: {res2['label']}")
    print(f"  --> Confidence: {res2['high_quality_probability'] * 100:.1f}%\n")
    print("=" * 60)


def run_interactive():
    print("=" * 60)
    print("      INTERACTIVE WINE QUALITY TESTER      ")
    print("   (Press Enter on any field to use the default typical value)   ")
    print("=" * 60)

    predictor = WineQualityPredictor()
    print(f"Loaded Trained Model: {predictor.model_name}\n")

    defaults = {
        "wine_type": (0, "Wine Type (0 for White, 1 for Red)"),
        "fixed acidity": (7.0, "Fixed Acidity (g/dm3)"),
        "volatile acidity": (0.28, "Volatile Acidity (g/dm3)"),
        "citric acid": (0.30, "Citric Acid (g/dm3)"),
        "residual sugar": (3.0, "Residual Sugar (g/dm3)"),
        "chlorides": (0.045, "Chlorides (g/dm3)"),
        "free sulfur dioxide": (30.0, "Free Sulfur Dioxide (mg/dm3)"),
        "total sulfur dioxide": (115.0, "Total Sulfur Dioxide (mg/dm3)"),
        "density": (0.9940, "Density (g/cm3)"),
        "pH": (3.20, "pH"),
        "sulphates": (0.50, "Sulphates (g/dm3)"),
        "alcohol": (11.5, "Alcohol (% vol)")
    }

    user_sample = {}
    for feature in predictor.feature_names:
        def_val, label = defaults[feature]
        val_str = input(f"{label} [default: {def_val}]: ").strip()
        if not val_str:
            user_sample[feature] = def_val
        else:
            try:
                user_sample[feature] = float(val_str)
            except ValueError:
                print(f"Invalid input, using default {def_val}")
                user_sample[feature] = def_val

    print("\nEvaluating wine characteristics...")
    result = predictor.predict_sample(user_sample)
    print("-" * 60)
    print(f"Prediction : {result['label']}")
    print(f"Confidence : {result['high_quality_probability'] * 100:.1f}%")
    print(f"Model      : {result['model_used']}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality Predictor")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run interactive terminal prompt")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo()


