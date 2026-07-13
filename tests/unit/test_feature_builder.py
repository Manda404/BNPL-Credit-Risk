from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL
from bnpl_credit_risk.exceptions import FeatureEngineeringError
from bnpl_credit_risk.features.builder import BNPLFeatureBuilder, resolve_feature_columns
from bnpl_credit_risk.features.transformers import FeatureEngineeringTransformer


def test_payment_stress_is_delay_times_missed(sample_df, project_config):
    out = BNPLFeatureBuilder(project_config.features).transform(sample_df)
    expected = sample_df["repayment_delay_days"] * sample_df["missed_payments"]
    assert (out["payment_stress"] == expected).all()


def test_income_to_purchase_ratio_no_division_by_zero(project_config):
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0],
            "purchase_amount": [0.0, -50.0],  # zero and (invalid) negative purchase amount
            "age": [30, 30],
            "transaction_date": ["2023-01-01", "2023-01-01"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [10.0, 10.0],
        }
    )
    out = BNPLFeatureBuilder(project_config.features).transform(df)
    assert out["income_to_purchase_ratio"].notna().all()
    assert (out["income_to_purchase_ratio"] < float("inf")).all()
    # purchase_amount clipped to >=0 then +1 floor => ratio == income / 1 for both rows
    assert out.loc[0, "income_to_purchase_ratio"] == pytest.approx(1000.0)
    assert out.loc[1, "income_to_purchase_ratio"] == pytest.approx(2000.0)


def test_age_group_out_of_range_maps_to_unknown(project_config):
    df = pd.DataFrame(
        {
            "age": [10, 30, 100],  # 10 and 100 are outside the configured [17,60] bins
            "monthly_income": [1000.0] * 3,
            "purchase_amount": [100.0] * 3,
            "transaction_date": ["2023-01-01"] * 3,
            "repayment_delay_days": [0] * 3,
            "missed_payments": [0] * 3,
            "risk_score": [10.0] * 3,
        }
    )
    out = BNPLFeatureBuilder(project_config.features).transform(df)
    assert out.loc[0, "age_group"] == "Unknown"
    assert out.loc[1, "age_group"] != "Unknown"
    assert out.loc[2, "age_group"] == "Unknown"


def test_txn_month_handles_unparseable_date(project_config):
    df = pd.DataFrame(
        {
            "age": [30, 30],
            "monthly_income": [1000.0, 1000.0],
            "purchase_amount": [100.0, 100.0],
            "transaction_date": ["2023-05-15", "not-a-date"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [10.0, 10.0],
        }
    )
    out = BNPLFeatureBuilder(project_config.features).transform(df)
    assert out.loc[0, "txn_month"] == 5
    assert out.loc[1, "txn_month"] == 0


def test_is_high_risk_threshold(project_config):
    threshold = project_config.features.engineered_features["is_high_risk"].risk_score_threshold
    df = pd.DataFrame(
        {
            "age": [30, 30],
            "monthly_income": [1000.0, 1000.0],
            "purchase_amount": [100.0, 100.0],
            "transaction_date": ["2023-01-01", "2023-01-01"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [threshold - 1, threshold + 1],
        }
    )
    out = BNPLFeatureBuilder(project_config.features).transform(df)
    assert out.loc[0, "is_high_risk"] == 0
    assert out.loc[1, "is_high_risk"] == 1


def test_missing_source_column_raises(project_config):
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(FeatureEngineeringError):
        BNPLFeatureBuilder(project_config.features).transform(df)


def test_resolve_feature_columns_excludes_leaky_for_application_risk(project_config):
    numeric, categorical = resolve_feature_columns(project_config.features, RISK_SCOPE_APPLICATION)
    leaky_raw = set(project_config.features.leaky_raw_columns)
    assert not (set(numeric) | set(categorical)) & leaky_raw
    assert "payment_stress" not in numeric
    assert "is_high_risk" not in numeric


def test_resolve_feature_columns_includes_behavioral_for_behavioral_risk(project_config):
    numeric, categorical = resolve_feature_columns(project_config.features, RISK_SCOPE_BEHAVIORAL)
    assert "missed_payments" in numeric
    assert "repayment_delay_days" in numeric
    assert "risk_score" in numeric
    assert "customer_segment" in categorical


def test_feature_engineering_transformer_is_sklearn_compatible(sample_df, project_config):
    transformer = FeatureEngineeringTransformer(project_config.features)
    fitted = transformer.fit(sample_df)
    assert fitted is transformer  # fit returns self, no learned state
    out = transformer.transform(sample_df)
    assert "payment_stress" in out.columns
