from __future__ import annotations

import numpy as np

from bnpl_credit_risk.evaluation.thresholding import ThresholdSelector, assign_risk_band
from bnpl_credit_risk.settings import CostMatrixConfig, RiskBandConfig, ThresholdPolicyConfig


def _config(**overrides) -> ThresholdPolicyConfig:
    base = {
        "policy": "best_f1",
        "fixed_value": 0.5,
        "min_recall": 0.6,
        "min_precision": 0.6,
        "cost_matrix": CostMatrixConfig(false_negative_cost=5.0, false_positive_cost=1.0),
    }
    base.update(overrides)
    return ThresholdPolicyConfig(**base)


def test_fixed_policy_returns_configured_value():
    selector = ThresholdSelector(_config(policy="fixed", fixed_value=0.42))
    result = selector.select(np.array([0, 1]), np.array([0.1, 0.9]))
    assert result["threshold"] == 0.42
    assert result["policy"] == "fixed"


def test_best_f1_finds_perfect_separation_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    selector = ThresholdSelector(_config(policy="best_f1"))
    result = selector.select(y_true, y_prob)
    y_pred = (y_prob >= result["threshold"]).astype(int)
    assert (y_pred == y_true).all()


def test_min_recall_respects_constraint():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.3, 0.6, 0.7, 0.9])
    selector = ThresholdSelector(_config(policy="min_recall", min_recall=0.75))
    result = selector.select(y_true, y_prob)
    y_pred = (y_prob >= result["threshold"]).astype(int)
    recall = (y_pred[y_true == 1] == 1).mean()
    assert recall >= 0.75


def test_cost_matrix_prefers_low_threshold_when_false_negatives_are_expensive():
    # Overlapping (non-perfectly-separable) probabilities so the optimal
    # threshold genuinely depends on the cost asymmetry, not just on ranking.
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8])
    expensive_fn = ThresholdSelector(
        _config(policy="cost_matrix", cost_matrix=CostMatrixConfig(false_negative_cost=50.0, false_positive_cost=1.0))
    ).select(y_true, y_prob)
    expensive_fp = ThresholdSelector(
        _config(policy="cost_matrix", cost_matrix=CostMatrixConfig(false_negative_cost=1.0, false_positive_cost=50.0))
    ).select(y_true, y_prob)
    assert expensive_fn["threshold"] < expensive_fp["threshold"]


def test_assign_risk_band_boundaries():
    bands = [
        RiskBandConfig(name="Low Risk", max_probability=0.25),
        RiskBandConfig(name="Medium Risk", max_probability=0.5),
        RiskBandConfig(name="High Risk", max_probability=0.75),
        RiskBandConfig(name="Very High Risk", max_probability=1.01),
    ]
    assert assign_risk_band(0.1, bands) == "Low Risk"
    assert assign_risk_band(0.25, bands) == "Low Risk"
    assert assign_risk_band(0.26, bands) == "Medium Risk"
    assert assign_risk_band(0.99, bands) == "Very High Risk"
