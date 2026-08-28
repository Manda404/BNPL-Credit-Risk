from __future__ import annotations

import pandas as pd

from bnpl_credit_risk.data.drift import DataDriftReport


def _report() -> DataDriftReport:
    return DataDriftReport(
        numeric_columns=("income",),
        categorical_columns=("segment",),
        target_column="target",
        date_column="date",
    )


def test_drift_report_marks_identical_distributions_stable():
    frame = pd.DataFrame(
        {
            "income": range(1, 101),
            "segment": ["A", "B"] * 50,
            "target": [0, 1] * 50,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )

    result = _report().build(frame, frame.copy())

    assert set(result["status"]) == {"stable"}
    assert float(result["psi"].max()) == 0.0


def test_drift_report_detects_unseen_categories():
    reference = pd.DataFrame(
        {
            "income": range(1, 101),
            "segment": ["A"] * 100,
            "target": [0, 1] * 50,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )
    current = reference.copy()
    current["segment"] = "NEW"

    result = _report().build(reference, current)
    segment = result.loc[result["feature"] == "segment"].iloc[0]

    assert segment["status"] == "critical"
    assert segment["unseen_category_rate"] == 1.0


def test_temporal_drift_uses_oldest_rows_as_reference():
    frame = pd.DataFrame(
        {
            "income": range(1, 101),
            "segment": ["A", "B"] * 50,
            "target": [0, 1] * 50,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )

    result = _report().build_temporal(frame, recent_fraction=0.20)

    assert len(result.reference) == 80
    assert len(result.current) == 20
    assert result.reference_end < result.current_start
