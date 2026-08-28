from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL
from bnpl_credit_risk.exceptions import FeatureEngineeringError
from bnpl_credit_risk.features.builder import BNPLFeatureBuilder, resolve_feature_columns
from bnpl_credit_risk.features.transformers import FeatureEngineeringTransformer


def test_payment_stress_is_delay_times_missed(sample_df, project_config):
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_BEHAVIORAL,
    ).transform(sample_df)
    expected = sample_df["repayment_delay_days"] * sample_df["missed_payments"]
    assert (out["payment_stress"] == expected).all()


def test_installment_burden_handles_zero_income_without_infinity(project_config):
    df = pd.DataFrame(
        {
            "monthly_income": [0.0, 2000.0],
            "purchase_amount": [600.0, -50.0],
            "bnpl_installments": [3, 3],
            "credit_score": [700, 700],
            "age": [30, 30],
            "transaction_date": ["2023-01-01", "2023-01-01"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [10.0, 10.0],
        }
    )
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_APPLICATION,
    ).transform(df)
    assert pd.isna(out.loc[0, "installment_burden_ratio"])
    assert out.loc[0, "affordability_band"] == "Unknown"
    assert out.loc[1, "installment_burden_ratio"] == pytest.approx(0.0)
    assert out.loc[1, "affordability_band"] == "Very Low"


def test_installment_affordability_features(project_config):
    df = pd.DataFrame(
        {
            "monthly_income": [2000.0],
            "purchase_amount": [600.0],
            "bnpl_installments": [3],
            "credit_score": [700],
            "age": [30],
            "transaction_date": ["2023-01-01"],
            "repayment_delay_days": [0],
            "missed_payments": [0],
            "risk_score": [10.0],
        }
    )
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_APPLICATION,
    ).transform(df)
    assert out.loc[0, "installment_amount"] == pytest.approx(200.0)
    assert out.loc[0, "installment_burden_ratio"] == pytest.approx(0.10)
    assert out.loc[0, "affordability_band"] == "High"
    assert out.loc[0, "income_after_installment"] == pytest.approx(1800.0)
    assert out.loc[0, "installment_term"] == "Short"
    assert out.loc[0, "credit_score_band"] == "Good"


def test_age_group_out_of_range_maps_to_unknown(project_config):
    df = pd.DataFrame(
        {
            "age": [10, 30, 101],
            "monthly_income": [1000.0] * 3,
            "purchase_amount": [100.0] * 3,
            "bnpl_installments": [3] * 3,
            "credit_score": [700] * 3,
            "transaction_date": ["2023-01-01"] * 3,
            "repayment_delay_days": [0] * 3,
            "missed_payments": [0] * 3,
            "risk_score": [10.0] * 3,
        }
    )
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_APPLICATION,
    ).transform(df)
    assert out.loc[0, "age_group"] == "Unknown"
    assert out.loc[1, "age_group"] != "Unknown"
    assert out.loc[2, "age_group"] == "Unknown"


def test_cyclical_month_handles_unparseable_date(project_config):
    df = pd.DataFrame(
        {
            "age": [30, 30],
            "monthly_income": [1000.0, 1000.0],
            "purchase_amount": [100.0, 100.0],
            "bnpl_installments": [3, 3],
            "credit_score": [700, 700],
            "transaction_date": ["2023-05-15", "not-a-date"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [10.0, 10.0],
        }
    )
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_APPLICATION,
    ).transform(df)
    assert out.loc[0, "transaction_month_sin"] == pytest.approx(0.5)
    assert out.loc[0, "transaction_month_cos"] == pytest.approx(-(3**0.5) / 2)
    assert out.loc[1, "transaction_month_sin"] == 0.0
    assert out.loc[1, "transaction_month_cos"] == 0.0


def test_is_high_risk_threshold(project_config):
    threshold = project_config.features.engineered_features["is_high_risk"].risk_score_threshold
    df = pd.DataFrame(
        {
            "age": [30, 30],
            "monthly_income": [1000.0, 1000.0],
            "purchase_amount": [100.0, 100.0],
            "bnpl_installments": [3, 3],
            "credit_score": [700, 700],
            "transaction_date": ["2023-01-01", "2023-01-01"],
            "repayment_delay_days": [0, 0],
            "missed_payments": [0, 0],
            "risk_score": [threshold - 1, threshold + 1],
        }
    )
    out = BNPLFeatureBuilder(
        project_config.features,
        risk_scope=RISK_SCOPE_BEHAVIORAL,
    ).transform(df)
    assert out.loc[0, "is_high_risk"] == 0
    assert out.loc[1, "is_high_risk"] == 1


def test_missing_source_column_raises(project_config):
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(FeatureEngineeringError):
        BNPLFeatureBuilder(
            project_config.features,
            risk_scope=RISK_SCOPE_APPLICATION,
        ).transform(df)


def test_resolve_feature_columns_excludes_leaky_for_application_risk(project_config):
    numeric, categorical = resolve_feature_columns(project_config.features, RISK_SCOPE_APPLICATION)
    leaky_raw = set(project_config.features.leaky_raw_columns)
    assert not (set(numeric) | set(categorical)) & leaky_raw
    assert "payment_stress" not in numeric
    assert "is_high_risk" not in numeric
    assert "installment_amount" in numeric
    assert "installment_burden_ratio" in numeric
    assert "income_after_installment" in numeric
    assert "credit_score_band" in categorical
    assert "affordability_band" in categorical
    assert "installment_term" in categorical


def test_resolve_feature_columns_includes_behavioral_for_behavioral_risk(project_config):
    numeric, categorical = resolve_feature_columns(project_config.features, RISK_SCOPE_BEHAVIORAL)
    assert "missed_payments" in numeric
    assert "repayment_delay_days" in numeric
    assert "risk_score" in numeric
    assert "customer_segment" in categorical


def test_feature_engineering_transformer_is_sklearn_compatible(sample_df, project_config):
    transformer = FeatureEngineeringTransformer(
        project_config.features,
        risk_scope=RISK_SCOPE_APPLICATION,
    )
    fitted = transformer.fit(sample_df)
    assert fitted is transformer  # fit returns self, no learned state
    out = transformer.transform(sample_df)
    assert "installment_amount" in out.columns
    assert "payment_stress" not in out.columns
    feature_names = transformer.get_feature_names_out(list(sample_df.columns))
    assert "installment_burden_ratio" in feature_names
    assert "payment_stress" not in feature_names
    assert "is_high_risk" not in feature_names
