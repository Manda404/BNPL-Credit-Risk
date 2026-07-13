"""Metric functions for a credit-risk binary classifier.

The notebook stopped at ROC-AUC + classification_report + confusion matrix.
A model whose output is used to grant or deny credit needs its probabilities
evaluated too (Brier score, log loss, calibration), and needs a
threshold-independent ranking metric that is robust to class imbalance
(PR-AUC), in addition to the usual precision/recall/F1 at the chosen
threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = confusion_counts(y_true, y_pred)
    denom = counts["true_negative"] + counts["false_positive"]
    return counts["true_negative"] / denom if denom else 0.0


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between TPR and FPR across thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def threshold_independent_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ks_statistic": ks_statistic(y_true, y_prob),
    }


def threshold_dependent_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_counts(y_true, y_pred),
    }


def probability_distribution_summary(y_prob: np.ndarray) -> dict[str, float]:
    series = pd.Series(y_prob)
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
    }


def metrics_by_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        row = {"threshold": float(t), **threshold_dependent_metrics(y_true, y_pred)}
        del row["confusion_matrix"]
        rows.append(row)
    return rows
