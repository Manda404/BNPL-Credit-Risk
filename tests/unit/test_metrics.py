from __future__ import annotations

import numpy as np

from bnpl_credit_risk.evaluation.metrics import (
    confusion_counts,
    ks_statistic,
    metrics_by_threshold,
    specificity_score,
    threshold_dependent_metrics,
    threshold_independent_metrics,
)


def test_confusion_counts_matches_manual_computation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 2}


def test_specificity_score_is_tn_rate():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert specificity_score(y_true, y_pred) == 2 / 3


def test_ks_statistic_is_one_for_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y_true, y_prob) == 1.0


def test_threshold_independent_metrics_keys_present():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    metrics = threshold_independent_metrics(y_true, y_prob)
    for key in ["roc_auc", "pr_auc", "brier_score", "log_loss", "ks_statistic"]:
        assert key in metrics


def test_threshold_dependent_metrics_confusion_matrix_present():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = threshold_dependent_metrics(y_true, y_pred)
    assert "confusion_matrix" in metrics
    assert metrics["recall"] == 1.0


def test_metrics_by_threshold_returns_one_row_per_threshold():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6, 0.3])
    thresholds = [0.2, 0.5, 0.8]
    rows = metrics_by_threshold(y_true, y_prob, thresholds)
    assert len(rows) == 3
    assert "confusion_matrix" not in rows[0]
