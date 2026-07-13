"""Evaluation plots: confusion matrix, ROC, precision-recall, feature
importance, probability distribution, metric-vs-threshold curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from bnpl_credit_risk.visualization.eda import COLORS, save_and_close


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path | None = None
) -> Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Paid", "Defaulted"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None) -> Figure:
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLORS["main"], linewidth=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["main"])
    ax.set_title("ROC Curve", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_confusion_matrix_and_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
    save_path: str | Path | None = None,
) -> Figure:
    """Confusion matrix + ROC curve side by side in one figure — the combined
    layout from the original notebook's evaluation cell, kept as a single
    view rather than two separate plots."""
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Paid", "Defaulted"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold", fontsize=12)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color=COLORS["main"], linewidth=2.5, label=f"{model_name} (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color=COLORS["main"])
    axes[1].set_title("ROC Curve", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None
) -> Figure:
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=COLORS["accent"], linewidth=2.5, label=f"AP = {ap:.3f}")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    model_name: str = "XGBoost",
    save_path: str | Path | None = None,
) -> Figure:
    """Matches the original notebook's feature importance plot: value labels
    next to each bar, a dashed "top 25%" threshold line, and bars colored red
    above that threshold."""
    top = importances.sort_values(ascending=True).tail(top_n)
    top_values = np.asarray(top.values, dtype=float)
    threshold = float(top.quantile(0.75))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = [COLORS["default"] if v > threshold else COLORS["main"] for v in top_values]
    bars = ax.barh(top.index, top_values, color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, top_values, strict=False):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5, label="Top 25% threshold")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_probability_distribution(y_prob: np.ndarray, save_path: str | Path | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_prob, bins=30, color=COLORS["main"], edgecolor="white", alpha=0.8)
    ax.set_title("Predicted Probability Distribution", fontweight="bold")
    ax.set_xlabel("P(default)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_metric_by_threshold(
    metrics_by_threshold: list[dict], metric_name: str = "f1_score", save_path: str | Path | None = None
) -> Figure:
    df = pd.DataFrame(metrics_by_threshold)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df[metric_name], color=COLORS["main"], linewidth=2)
    ax.set_title(f"{metric_name} by threshold", fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric_name)
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
