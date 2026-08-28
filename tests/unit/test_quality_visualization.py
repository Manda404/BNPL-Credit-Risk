from __future__ import annotations

import matplotlib.pyplot as plt

from bnpl_credit_risk.data.quality import DataQualityReport
from bnpl_credit_risk.visualization.quality import plot_data_quality_report


def test_plot_data_quality_report_builds_dashboard(tmp_path):
    report = {
        "n_rows": 100,
        "n_columns": 4,
        "n_duplicate_rows": 2,
        "n_missing_values_total": 3,
        "missing_values_by_column": {"income": 3},
        "dtypes": {"user_id": "int64", "income": "float64", "segment": "object"},
        "target_distribution": {
            "0": {"count": 60, "rate": 0.6},
            "1": {"count": 40, "rate": 0.4},
        },
    }
    output_path = tmp_path / "quality.png"

    figure = plot_data_quality_report(report, save_path=output_path, size=(15, 8))

    assert output_path.exists()
    assert len(figure.axes) == 4
    assert tuple(figure.get_size_inches()) == (15.0, 8.0)
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Data Quality Report"
    plt.close(figure)


def test_plot_data_quality_report_handles_clean_report():
    report = {
        "n_rows": 10,
        "n_columns": 2,
        "n_duplicate_rows": 0,
        "n_missing_values_total": 0,
        "missing_values_by_column": {},
        "dtypes": {"user_id": "int64", "value": "float64"},
    }

    figure = plot_data_quality_report(report)

    assert len(figure.axes) == 4
    assert any(text.get_text() == "No missing values" for text in figure.axes[1].texts)
    plt.close(figure)


def test_data_quality_report_show_exposes_notebook_facade(sample_df, monkeypatch):
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    result = DataQualityReport(target_column="default_flag").show(sample_df)

    assert result is None
    assert show_calls == [True]
    plt.close("all")
