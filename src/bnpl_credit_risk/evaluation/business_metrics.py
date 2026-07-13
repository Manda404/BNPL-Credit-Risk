"""Business-cost view of a classifier's decisions: converts confusion-matrix
counts into a currency-denominated cost using configs/training.yaml's
cost_matrix, and reports how cost trades off against threshold choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bnpl_credit_risk.evaluation.metrics import confusion_counts
from bnpl_credit_risk.settings import CostMatrixConfig


def business_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: CostMatrixConfig) -> dict[str, Any]:
    counts = confusion_counts(y_true, y_pred)
    fn_total = counts["false_negative"] * cost_matrix.false_negative_cost
    fp_total = counts["false_positive"] * cost_matrix.false_positive_cost
    total_cost = fn_total + fp_total
    return {
        "false_negative_cost_total": fn_total,
        "false_positive_cost_total": fp_total,
        "total_cost": total_cost,
        "average_cost_per_decision": total_cost / len(y_true) if len(y_true) else 0.0,
        "confusion_matrix": counts,
    }


def cost_by_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, cost_matrix: CostMatrixConfig, thresholds: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = business_cost(y_true, y_pred, cost_matrix)
        rows.append({"threshold": float(t), "total_cost": cost["total_cost"]})
    return rows
