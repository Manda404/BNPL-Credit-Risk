from __future__ import annotations

import pandas as pd

from bnpl_credit_risk.pipelines.data_split_pipeline import StratifiedDataSplitPipeline


def test_data_split_pipeline_creates_csv_files_and_figure(test_settings, test_config):
    source_df = pd.read_csv(test_settings.resolve(test_settings.data_raw_path))

    pipeline = StratifiedDataSplitPipeline(
        settings=test_settings,
        config=test_config,
    )
    result = pipeline.run(source_df)

    assert result.paths.train_path.exists()
    assert result.paths.test_path.exists()
    assert result.figure_path.exists()
    assert tuple(result.figure.get_size_inches()) == tuple(
        test_config.data_split.visualization.size
    )
    train_df = pd.read_csv(result.paths.train_path)
    test_df = pd.read_csv(result.paths.test_path)
    assert len(train_df) == 180
    assert len(test_df) == 20
    assert abs(train_df["default_flag"].mean() - test_df["default_flag"].mean()) < 0.01


def test_data_split_pipeline_accepts_prepared_application_features(test_settings, test_config):
    source_df = pd.read_csv(test_settings.resolve(test_settings.data_raw_path))
    source_df = source_df.drop(columns=list(test_config.features.leaky_raw_columns))

    for feature_name in (
        "installment_amount",
        "installment_burden_ratio",
        "income_after_installment",
        "transaction_month_sin",
        "transaction_month_cos",
    ):
        source_df[feature_name] = 0.0
    for feature_name in (
        "age_group",
        "credit_score_band",
        "affordability_band",
        "installment_term",
    ):
        source_df[feature_name] = "placeholder"

    result = StratifiedDataSplitPipeline(
        settings=test_settings,
        config=test_config,
    ).run(source_df, input_stage="prepared")

    assert len(result.split.train) == 180
    assert len(result.split.test) == 20
