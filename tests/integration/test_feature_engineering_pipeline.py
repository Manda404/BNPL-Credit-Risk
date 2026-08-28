from __future__ import annotations

import pandas as pd

from bnpl_credit_risk.features.leakage import BNPLLeakageGuard
from bnpl_credit_risk.pipelines.feature_engineering_pipeline import (
    FeatureEngineeringPipeline,
)


def test_feature_pipeline_materializes_identical_train_test_contract(
    sample_df,
    test_config,
    test_settings,
):
    safe = BNPLLeakageGuard(
        test_config.features,
        test_config.model.risk_scope,
    ).apply(sample_df).cleaned

    result = FeatureEngineeringPipeline(
        settings=test_settings,
        config=test_config,
    ).run(safe.iloc[:180].copy(), safe.iloc[180:].copy())

    assert list(result.train.columns) == list(result.test.columns)
    assert "installment_burden_ratio" in result.train.columns
    assert "affordability_band" in result.train.columns
    assert "transaction_month_sin" in result.train.columns
    assert result.paths.train_path.exists()
    assert result.paths.test_path.exists()
    assert list(pd.read_csv(result.paths.train_path).columns) == list(result.train.columns)
