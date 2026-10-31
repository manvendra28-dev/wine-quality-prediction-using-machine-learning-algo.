# 🍷 Wine Quality Prediction Model Using Machine Learning

An end-to-end, production-grade Machine Learning solution to evaluate, predict, and assure wine quality using physicochemical laboratory measurements.

Based on the renowned **UCI Wine Quality Dataset** (Cortez et al.), encompassing both Red Wine (1,599 samples) and White Wine (4,898 samples).

---

## 📌 Project Overview & Key Results

- **Dataset**: 6,497 total samples (1,599 Red Wine, 4,898 White Wine) with 11 physicochemical attributes + wine type.
- **Problem Formulation**: High Quality wine classification ($\text{Quality} \ge 7$) vs. Normal/Low Quality ($\text{Quality} < 7$).
- **Data Integrity**: Clean 80/20 stratified split performed *before* fitting any transformer or model to prevent data leakage.
- **Champion Model**: **Histogram-based Gradient Boosting Classifier** (`HistGradientBoostingClassifier`).
  - **Accuracy**: **88.31%**
  - **ROC-AUC**: **90.09%**
  - **Precision**: **75.49%**
  - **F1-Score**: **0.67**
  - **Inference Latency**: Fast sub-millisecond per sample.

---

## 📊 Cross-Validation Leaderboard (5-Fold Stratified CV)

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-Score | CV ROC-AUC | Fit Time (s) |
|---|---|---|---|---|---|---|
| **Hist Gradient Boosting** | **87.49%** | **73.73%** | **56.41%** | **0.6384** | **0.8938** | **0.71s** |
| Random Forest | 88.09% | 79.25% | 53.28% | 0.6367 | 0.9104 | 7.81s |
| Gradient Boosting | 86.32% | 70.75% | 51.81% | 0.5976 | 0.8904 | 3.77s |
| Extra Trees | 86.49% | 81.81% | 40.15% | 0.5384 | 0.9055 | 1.42s |
| Logistic Regression (Linear Baseline) | 81.60% | 57.12% | 26.74% | 0.3630 | 0.8112 | 6.75s |
| Dummy (Majority Baseline) | 80.35% | 0.00% | 0.00% | 0.0000 | 0.5000 | 9.21s |

---

## 🔬 Key Physicochemical Drivers (Permutation Importance)

1. **Alcohol (0.057)**: Strongest positive determinant. Higher alcohol content consistently corresponds to premium sensory ratings.
2. **Volatile Acidity (0.030)**: Strong negative determinant. High acetic acid gives a vinegar-like defect, sharply degrading quality.
3. **Free Sulfur Dioxide (0.029)**: Critical antioxidant preventing oxidation and browning.
4. **Residual Sugar (0.029)** & **Sulphates (0.027)**: Essential for mouthfeel, balance, and antimicrobial protection.

---

## 📁 Repository Structure

```text
├── data/
│   ├── winequality-red.csv          # UCI Red Wine dataset (1,599 rows)
│   └── winequality-white.csv        # UCI White Wine dataset (4,898 rows)
├── models/
│   └── best_wine_model_combined.joblib  # Serialized best model pipeline
├── notebooks/
│   └── wine_quality_analysis.ipynb  # Interactive step-by-step notebook
├── reports/
│   ├── cv_model_comparison.csv      # Cross-validation performance table
│   ├── feature_importance.csv       # Permutation importance rankings
│   ├── test_predictions.csv         # Out-of-sample predictions
│   └── figures/
│       ├── model_comparison.png     # Algorithm comparison chart
│       ├── confusion_matrix.png     # Confusion matrix heatmap
│       ├── roc_curve.png            # ROC-AUC curve
│       ├── feature_importance.png   # Ranked feature importance chart
│       └── correlation_matrix.png   # Physicochemical correlation heatmap
├── src/
│   ├── data_loader.py               # Ingestion, validation, stratified splitting
│   ├── train.py                     # Pipeline training & CV benchmarking
│   ├── evaluate.py                  # Report & visualization generation
│   └── predict.py                   # Inference CLI & sample tester
├── download_data.py                 # Automated dataset downloader
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Data
```bash
python download_data.py
```

### 3. Train & Benchmark Models
```bash
python src/train.py
```

### 4. Generate Visual Evaluation Reports
```bash
python src/evaluate.py
```

### 5. Run Desktop GUI (Tkinter)
```bash
python gui_app.py
```

### 6. Interactive Slide Deck Presentation
- Open `presentation.html` in your browser for an animated slide deck.
- Or view `presentation_slides.md` for speaker notes and slides.

### 7. Run CLI Predictions
```bash
python src/predict.py
# Or interactive mode:
python src/predict.py -i
```

Sample output:
```text
[Sample 1: High-tier White Wine]
  --> Prediction: High Quality (Score >= 7)
  --> Confidence: 70.1%

[Sample 2: Low-tier Red Wine]
  --> Prediction: Normal / Low Quality (Score < 7)
  --> Confidence: 0.0%
```

### 6. Interactive Notebook
Open `notebooks/wine_quality_analysis.ipynb` in VS Code or Jupyter to explore the interactive analysis and visualizations.

