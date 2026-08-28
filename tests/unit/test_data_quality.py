from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bnpl_credit_risk.data.profiling import DatasetProfileOptions
from bnpl_credit_risk.data.quality import DataQualityReport


def test_data_quality_report_basic_stats(sample_df):
    report = DataQualityReport(target_column="default_flag").build(sample_df)

    assert report["n_rows"] == 200
    assert "target_distribution" in report
    assert set(report["target_distribution"]) == {"0", "1"}


def test_data_quality_report_profiles_columns_and_iqr_outliers():
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 100.0, np.nan],
            "segment": ["A", "A", "B", None, "C", "A"],
        }
    )

    profile = DataQualityReport().profile_dataset(df)
    amount = profile.loc[profile["Column"] == "amount"].iloc[0]
    segment = profile.loc[profile["Column"] == "segment"].iloc[0]

    assert list(profile.columns[:6]) == [
        "Column",
        "Type",
        "Non-Null",
        "Missing",
        "% Missing",
        "Cardinality",
    ]
    assert amount["Missing"] == 1
    assert amount["% Missing"] == 16.67
    assert amount["Cardinality"] == 5
    assert amount["Outliers"] == 1
    assert amount["% Outliers"] == 20.0
    assert amount["Max"] == 100.0
    assert len(amount["Examples"]) == 5
    assert pd.isna(segment["Outliers"])
    assert not {
        "Skewness",
        "Top Value",
        "Top Frequency",
        "Memory (KB)",
    }.intersection(profile.columns)


def test_data_quality_profile_handles_empty_dataframe():
    profile = DataQualityReport().profile_dataset(pd.DataFrame(columns=["empty"]))

    assert len(profile) == 1
    assert profile.loc[0, "Non-Null"] == 0
    assert profile.loc[0, "Missing"] == 0
    assert profile.loc[0, "% Missing"] == 0.0
    assert profile.loc[0, "Examples"] == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numeric_examples": 0}, "Example limits"),
        ({"categorical_examples": 0}, "Example limits"),
        ({"iqr_multiplier": 0}, "iqr_multiplier"),
    ],
)
def test_profile_options_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DatasetProfileOptions(**kwargs)
