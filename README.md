# Coffee Quality Risk Classification

A machine learning project that classifies Arabica coffee samples as **Good** or **Defective** based on sensory and physical attributes.

> ATW306 Group Project

---

## Overview

This project builds and compares five classification models to predict whether a coffee sample is defective. It covers the full ML pipeline: data loading, exploratory data analysis (EDA), model training with cross-validation, evaluation, and visualization.

**Target variable:** A sample is labelled *Defective* if it has any Category One or Category Two defects; otherwise it is labelled *Good*.

---

## Dataset

`Arabica.csv` — Arabica coffee quality records from the [Coffee Quality Institute (CQI)](https://github.com/jldbc/coffee-quality-database).

**Features used:**

| Feature | Description |
|---|---|
| Aroma | Fragrance/aroma score |
| Flavor | Flavor score |
| Aftertaste | Aftertaste score |
| Acidity | Acidity score |
| Body | Body score |
| Balance | Balance score |
| Uniformity | Uniformity score |
| Clean Cup | Clean cup score |
| Sweetness | Sweetness score |
| Moisture Percentage | Moisture content (%) |

---

## Models

Five classifiers are trained and compared:

- Logistic Regression
- Random Forest (300 estimators)
- Support Vector Machine (SVM)
- Gradient Boosting (200 estimators)
- K-Nearest Neighbors (KNN)

All models are evaluated with **5-fold stratified cross-validation** and tested on a held-out 20% test set.

---

## Project Structure

```
coffee-quality-project-main/
├── coffee_quality.py       # Main script
├── Arabica.csv             # Dataset
└── outputs/                # Generated figures and predictions
    ├── 01_class_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_distributions.png
    ├── 04_model_comparison.png
    ├── 05_roc_curves.png
    ├── 06_confusion_matrices.png
    ├── 07_feature_importance.png
    ├── 08_permutation_importance.png
    └── sample_predictions.csv
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

```bash
python coffee_quality.py
```

All output charts and the prediction CSV will be saved to the `outputs/` folder.

---

## Output Visualizations

| File | Description |
|---|---|
| `01_class_distribution.png` | Bar chart and pie chart of Good vs Defective samples |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_feature_distributions.png` | Histogram of each feature split by class |
| `04_model_comparison.png` | Accuracy / Precision / Recall / F1 bar chart for all models |
| `05_roc_curves.png` | Overlaid ROC curves with AUC scores |
| `06_confusion_matrices.png` | Confusion matrix for each model |
| `07_feature_importance.png` | Random Forest built-in feature importance |
| `08_permutation_importance.png` | Permutation importance with standard deviation |
| `sample_predictions.csv` | First 50 test-set predictions with probabilities |

---

## Results

The script prints a final summary to the console after training:

```
=======================================================
          FINAL RESULTS SUMMARY (Test Set)
=======================================================
  Logistic Regression    Acc=X.XXXX  F1=X.XXXX
  Random Forest          Acc=X.XXXX  F1=X.XXXX
  SVM                    Acc=X.XXXX  F1=X.XXXX
  Gradient Boosting      Acc=X.XXXX  F1=X.XXXX
  KNN                    Acc=X.XXXX  F1=X.XXXX

  → Best model by CV F1: <model name>
=======================================================
```

---

## Dependencies

| Package | Role |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Models, preprocessing, evaluation |
| `matplotlib` | Plotting |
| `seaborn` | Heatmap visualization |
