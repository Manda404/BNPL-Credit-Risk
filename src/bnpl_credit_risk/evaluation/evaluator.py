"""ClassificationEvaluator — orchestrates every metric in evaluation/metrics.py
plus business_metrics.py into one report dict, given true labels, predicted
probabilities and the selected decision threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from bnpl_credit_risk.evaluation.business_metrics import business_cost
from bnpl_credit_risk.evaluation.metrics import (
    metrics_by_threshold,
    probability_distribution_summary,
    threshold_dependent_metrics,
    threshold_independent_metrics,
)
from bnpl_credit_risk.settings import CostMatrixConfig


class ClassificationEvaluator:
    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        *,
        cost_matrix: CostMatrixConfig | None = None,
        threshold_grid: list[float] | None = None,
    ) -> dict[str, Any]:
        y_pred = (y_prob >= threshold).astype(int)

        report: dict[str, Any] = {
            "threshold_used": float(threshold),
            "n_samples": int(len(y_true)),
            "positive_rate_actual": float(np.mean(y_true)),
            "positive_rate_predicted": float(np.mean(y_pred)),
            **threshold_independent_metrics(y_true, y_prob),
            **threshold_dependent_metrics(y_true, y_pred),
            "probability_distribution": probability_distribution_summary(y_prob),
        }

        if threshold_grid:
            report["metrics_by_threshold"] = metrics_by_threshold(y_true, y_prob, threshold_grid)

        if cost_matrix is not None:
            report["business_cost"] = business_cost(y_true, y_pred, cost_matrix)

        logger.bind(pipeline="evaluation.evaluator").info(
            "Evaluation: roc_auc={:.4f} pr_auc={:.4f} brier={:.4f} f1={:.4f} threshold={:.3f}",
            report["roc_auc"],
            report["pr_auc"],
            report["brier_score"],
            report["f1_score"],
            threshold,
        )
        return report
