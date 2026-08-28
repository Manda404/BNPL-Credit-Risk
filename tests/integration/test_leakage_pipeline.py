from __future__ import annotations

import pandas as pd

from bnpl_credit_risk.pipelines.leakage_pipeline import LeakageControlPipeline


def test_leakage_pipeline_materializes_safe_partitions(sample_df, test_config, test_settings):
    train_df = sample_df.iloc[:180].copy()
    test_df = sample_df.iloc[180:].copy()

    result = LeakageControlPipeline(
        settings=test_settings,
        config=test_config,
    ).run(train_df, test_df)

    leaky_columns = set(test_config.features.leaky_raw_columns)
    assert not leaky_columns.intersection(result.train.columns)
    assert not leaky_columns.intersection(result.test.columns)
    assert result.paths.train_path.exists()
    assert result.paths.test_path.exists()
    assert list(pd.read_csv(result.paths.train_path).columns) == list(result.train.columns)
