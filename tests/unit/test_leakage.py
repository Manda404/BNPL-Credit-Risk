from __future__ import annotations

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL
from bnpl_credit_risk.features.leakage import BNPLLeakageGuard


def test_application_guard_removes_configured_leakage_columns(sample_df, project_config):
    result = BNPLLeakageGuard(
        project_config.features,
        RISK_SCOPE_APPLICATION,
    ).apply(sample_df)

    assert set(result.removed_columns) == set(project_config.features.leaky_raw_columns)
    assert not set(project_config.features.leaky_raw_columns).intersection(result.cleaned.columns)
    assert set(result.report["Decision"]) == {"Remove"}


def test_behavioral_guard_retains_post_origination_columns(sample_df, project_config):
    result = BNPLLeakageGuard(
        project_config.features,
        RISK_SCOPE_BEHAVIORAL,
    ).apply(sample_df)

    assert result.removed_columns == ()
    assert set(project_config.features.leaky_raw_columns).issubset(result.cleaned.columns)
