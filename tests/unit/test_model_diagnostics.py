from __future__ import annotations

import numpy as np

from bnpl_credit_risk.evaluation.diagnostics import (
    ThresholdTradeoffAnalyzer,
    calibration_table,
    expected_calibration_error,
    lift_and_gains_table,
)
from bnpl_credit_risk.settings import CostMatrixConfig


def test_threshold_tradeoff_marks_selected_threshold_and_business_counts():
    analyzer = ThresholdTradeoffAnalyzer(
        CostMatrixConfig(false_negative_cost=5.0, false_positive_cost=1.0)
    )
    table = analyzer.build(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.6, 0.4, 0.9]),
        np.array([0.3, 0.7]),
        selected_threshold=0.5,
    )

    selected = table.loc[table["Selected"]].iloc[0]
    assert selected["Defaults caught"] == 1
    assert selected["Defaults missed"] == 1
    assert selected["False positive alerts"] == 1
    assert selected["Estimated cost"] == 6.0


def test_calibration_error_is_zero_for_matching_bin_rates():
    table = calibration_table(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        n_bins=2,
    )

    assert expected_calibration_error(table) == 0.0


def test_highest_risk_group_has_positive_lift_for_ordered_scores():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    probability = np.linspace(0.95, 0.05, len(y_true))

    table = lift_and_gains_table(y_true, probability, n_groups=5)

    assert table.loc[0, "Risk group"] == 1
    assert table.loc[0, "Lift"] > 1.0
    assert table.loc[0, "Cumulative_defaults_caught"] > table.loc[
        0, "Cumulative_population"
    ]
