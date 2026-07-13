"""ThresholdSelector — chooses a decision threshold instead of assuming 0.5.

Selection happens on out-of-fold probabilities from the TRAIN split (see
pipelines/training_pipeline.py), never on the held-out test set — otherwise
the reported test metrics would be optimistic because the threshold was
tuned on the same data used to evaluate it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from bnpl_credit_risk.constants import (
    THRESHOLD_BEST_F1,
    THRESHOLD_COST_MATRIX,
    THRESHOLD_FIXED,
    THRESHOLD_MIN_PRECISION,
    THRESHOLD_MIN_RECALL,
)
from bnpl_credit_risk.evaluation.metrics import confusion_counts
from bnpl_credit_risk.exceptions import ConfigError
from bnpl_credit_risk.settings import RiskBandConfig, ThresholdPolicyConfig


def assign_risk_band(probability: float, risk_bands: list[RiskBandConfig]) -> str:
    """Map a probability to the first configured band whose max_probability it falls under.

    Bands must be supplied sorted ascending by max_probability (as in
    configs/training.yaml); the last band's max_probability should be >1.0 so
    every probability in [0, 1] is covered.
    """
    for band in risk_bands:
        if probability <= band.max_probability:
            return band.name
    return risk_bands[-1].name if risk_bands else "Unknown"


class ThresholdSelector:
    def __init__(self, config: ThresholdPolicyConfig) -> None:
        self._config = config

    def select(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        policy = self._config.policy
        if policy == THRESHOLD_FIXED:
            threshold = self._config.fixed_value
            rationale = "Fixed threshold from configs/training.yaml (not tuned on data)."
        elif policy == THRESHOLD_BEST_F1:
            threshold, rationale = self._best_f1(y_true, y_prob)
        elif policy == THRESHOLD_MIN_RECALL:
            threshold, rationale = self._min_recall(y_true, y_prob)
        elif policy == THRESHOLD_MIN_PRECISION:
            threshold, rationale = self._min_precision(y_true, y_prob)
        elif policy == THRESHOLD_COST_MATRIX:
            threshold, rationale = self._cost_matrix(y_true, y_prob)
        else:
            raise ConfigError(f"Unknown threshold policy: {policy}")

        return {"threshold": float(threshold), "policy": policy, "rationale": rationale}

    def _best_f1(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0
        )
        # precision_recall_curve returns one more point than thresholds (last point has no threshold)
        best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
        threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
        return threshold, f"Threshold maximizing F1 on out-of-fold train predictions (F1={f1[best_idx]:.4f})."

    def _min_recall(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        eligible = recall[:-1] >= self._config.min_recall
        if not eligible.any():
            return 0.0, f"No threshold reaches recall>={self._config.min_recall}; defaulted to 0.0 (flag everyone)."
        candidate_idx = np.where(eligible)[0]
        best_idx = candidate_idx[int(np.argmax(precision[candidate_idx]))]
        return float(thresholds[best_idx]), (
            f"Highest-precision threshold subject to recall>={self._config.min_recall} "
            f"(precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f})."
        )

    def _min_precision(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        eligible = precision[:-1] >= self._config.min_precision
        if not eligible.any():
            return 1.0, f"No threshold reaches precision>={self._config.min_precision}; defaulted to 1.0 (flag no one)."
        candidate_idx = np.where(eligible)[0]
        best_idx = candidate_idx[int(np.argmax(recall[candidate_idx]))]
        return float(thresholds[best_idx]), (
            f"Highest-recall threshold subject to precision>={self._config.min_precision} "
            f"(precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f})."
        )

    def _cost_matrix(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
        fn_cost = self._config.cost_matrix.false_negative_cost
        fp_cost = self._config.cost_matrix.false_positive_cost
        candidates = np.linspace(0.01, 0.99, 99)
        costs = []
        for t in candidates:
            y_pred = (y_prob >= t).astype(int)
            counts = confusion_counts(y_true, y_pred)
            cost = counts["false_negative"] * fn_cost + counts["false_positive"] * fp_cost
            costs.append(cost)
        best_idx = int(np.argmin(costs))
        return float(candidates[best_idx]), (
            f"Threshold minimizing total cost (FN cost={fn_cost}, FP cost={fp_cost}); "
            f"total cost={costs[best_idx]:.2f}."
        )
