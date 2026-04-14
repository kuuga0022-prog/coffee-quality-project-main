"""
Coffee Quality Risk Classification 
ATW306 Group Project

Improvements over baseline:
- 5 models compared (LR, RF, SVM, Gradient Boosting, KNN)
- 5-fold stratified cross-validation
- ROC curves for all models (overlaid)
- Model performance summary chart (Accuracy / F1 / Precision / Recall)
- Correlation heatmap
- Feature distribution plots by class
- RF Feature Importance + Permutation Importance

Output files (outputs/ folder):
    01_class_distribution.png
    02_correlation_heatmap.png
    03_feature_distributions.png
    04_model_comparison.png
    05_roc_curves.png
    06_confusion_matrices.png
    07_feature_importance.png
    08_permutation_importance.png
    sample_predictions.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = "Arabica.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = {
    "good": "#4CAF50",
    "defective": "#E53935",
    "bg": "#FAFAFA",
}

FEATURE_COLS = [
    "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
    "Balance", "Uniformity", "Clean Cup", "Sweetness",
    "Moisture Percentage",
]

# ─── 1. Data Loading & Preprocessing ───────────────────────────────────────────
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}.")
    df = pd.read_csv(path, encoding="latin1")

    keep = FEATURE_COLS + ["Category One Defects", "Category Two Defects"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[keep].dropna().reset_index(drop=True)

    # Target: Defective if any defect > 0
    df["defect"] = (
        (df["Category One Defects"] > 0) | (df["Category Two Defects"] > 0)
    ).astype(int)
    df = df.drop(["Category One Defects", "Category Two Defects"], axis=1)

    print(f"Dataset ready: {df.shape[0]} samples, {df.shape[1]-1} features")
    print("\nClass distribution:")
    vc = df["defect"].value_counts()
    for k, v in vc.items():
        label = "Good" if k == 0 else "Defective"
        print(f"  {label}: {v} ({v/len(df)*100:.1f}%)")
    return df


# ─── 2. EDA Visualisations ─────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["defect"].value_counts().sort_index()
    labels = ["Good", "Defective"]
    colors = [PALETTE["good"], PALETTE["defective"]]

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.suptitle("Target Variable Overview", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 01_class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 02_correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = FEATURE_COLS
    n_cols = 5
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for cls, color, label in [(0, PALETTE["good"], "Good"), (1, PALETTE["defective"], "Defective")]:
            subset = df[df["defect"] == cls][feat]
            ax.hist(subset, bins=15, alpha=0.6, color=color, label=label, edgecolor="white")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions: Good vs Defective", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 03_feature_distributions.png")


# ─── 3. Model Definitions ──────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=10,
                                                      min_samples_split=5, min_samples_leaf=2,
                                                      random_state=42, class_weight="balanced"),
        "SVM":                 SVC(probability=True, class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }


# ─── 4. Cross-Validation ───────────────────────────────────────────────────────
def run_cross_validation(models: dict, X: np.ndarray, y: pd.Series) -> pd.DataFrame:
    print("\n--- 5-Fold Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    records = []

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        records.append({
            "Model":     name,
            "Accuracy":  scores["test_accuracy"].mean(),
            "Precision": scores["test_precision_weighted"].mean(),
            "Recall":    scores["test_recall_weighted"].mean(),
            "F1":        scores["test_f1_weighted"].mean(),
            "Acc Std":   scores["test_accuracy"].std(),
        })
        print(f"  {name:<22} Acc={records[-1]['Accuracy']:.4f} ± {records[-1]['Acc Std']:.4f}  F1={records[-1]['F1']:.4f}")

    return pd.DataFrame(records)


# ─── 5. Model Comparison Chart ─────────────────────────────────────────────────
def plot_model_comparison(cv_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(cv_df))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, cv_df[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(cv_df["Model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (5-Fold CV Mean)", fontsize=11)
    ax.set_title("Model Comparison — Accuracy / Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 04_model_comparison.png")


# ─── 6. ROC Curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(models: dict, X_train, X_test, y_train, y_test) -> dict:
    line_colors = ["#2196F3", "#E53935", "#4CAF50", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 6))
    trained = {}

    for (name, model), color in zip(models.items(), line_colors):
        model.fit(X_train, y_train)
        trained[name] = model

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 05_roc_curves.png")
    return trained


# ─── 7. Confusion Matrices ─────────────────────────────────────────────────────
def plot_confusion_matrices(trained: dict, X_test, y_test) -> None:
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Defective"])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(name, fontsize=9, fontweight="bold")

    plt.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 06_confusion_matrices.png")


# ─── 8. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names) -> None:
    importance = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    colors = ["#E53935" if v == importance.max() else "#2196F3" for v in importance]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 07_feature_importance.png")


def plot_permutation_importance(rf_model, X_test, y_test, feature_names) -> None:
    result = permutation_importance(rf_model, X_test, y_test, n_repeats=20, random_state=42)
    perm = pd.Series(result.importances_mean, index=feature_names).sort_values()
    errors = pd.Series(result.importances_std, index=feature_names).reindex(perm.index)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm.index, perm.values, xerr=errors.values,
            color="#4CAF50", edgecolor="white", error_kw={"elinewidth": 1.5, "capsize": 4})
    ax.set_xlabel("Mean Accuracy Decrease", fontsize=11)
    ax.set_title("Random Forest — Permutation Importance (±Std)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 08_permutation_importance.png")


# ─── 9. Save Predictions ───────────────────────────────────────────────────────
def save_predictions(trained: dict, X_test, y_test, feature_names) -> None:
    rf = trained["Random Forest"]
    preds = pd.DataFrame(X_test, columns=feature_names)
    preds["actual"] = y_test.values
    preds["predicted"] = rf.predict(X_test)
    preds["predicted_label"] = preds["predicted"].map({0: "Good", 1: "Defective"})
    preds["defect_probability"] = rf.predict_proba(X_test)[:, 1].round(4)
    preds.head(50).to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
    print("Saved: sample_predictions.csv")


# ─── 10. Print Final Summary ───────────────────────────────────────────────────
def print_summary(trained: dict, X_test, y_test, cv_df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("          FINAL RESULTS SUMMARY (Test Set)")
    print("=" * 55)
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0, average="weighted")
        print(f"  {name:<22}  Acc={acc:.4f}  F1={f1:.4f}")

    best = cv_df.loc[cv_df["F1"].idxmax(), "Model"]
    print(f"\n  → Best model by CV F1: {best}")
    print("=" * 55)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        print("=" * 55)
        print("   Coffee Quality Classification — Enhanced")
        print("=" * 55)

        df = load_and_prepare(DATA_FILE)

        print("\n[EDA] Generating visualisations...")
        plot_class_distribution(df)
        plot_correlation_heatmap(df)
        plot_feature_distributions(df)

        X_raw = df.drop("defect", axis=1)
        y     = df["defect"]
        feature_names = X_raw.columns.tolist()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        models = get_models()

        print("\n[CV] Running cross-validation...")
        cv_df = run_cross_validation(models, X_train, y_train)
        plot_model_comparison(cv_df)

        print("\n[Models] Training on full train set + plotting ROC...")
        models_fresh = get_models()
        trained = plot_roc_curves(models_fresh, X_train, X_test, y_train, y_test)

        plot_confusion_matrices(trained, X_test, y_test)

        rf = trained["Random Forest"]
        plot_feature_importance(rf, feature_names)
        plot_permutation_importance(rf, X_test, y_test, feature_names)

        save_predictions(trained, X_test, y_test, feature_names)
        print_summary(trained, X_test, y_test, cv_df)

        print(f"\nAll outputs saved to '{OUTPUT_DIR}/'")

    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
