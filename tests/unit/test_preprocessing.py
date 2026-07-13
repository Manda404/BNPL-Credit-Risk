from __future__ import annotations

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION
from bnpl_credit_risk.features.feature_names import get_output_feature_names
from bnpl_credit_risk.features.preprocessing import build_preprocessing_pipeline


def test_preprocessing_pipeline_fits_and_transforms(sample_df, project_config):
    target = project_config.data.target_column
    X = sample_df.drop(columns=[target])
    pipeline = build_preprocessing_pipeline(project_config.features, RISK_SCOPE_APPLICATION)
    transformed = pipeline.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_preprocessing_handles_unknown_category_at_transform_time(sample_df, project_config):
    target = project_config.data.target_column
    X = sample_df.drop(columns=[target])
    train_X = X.iloc[:150]
    test_X = X.iloc[150:].copy()
    test_X["employment_type"] = "NeverSeenBefore"

    pipeline = build_preprocessing_pipeline(project_config.features, RISK_SCOPE_APPLICATION)
    pipeline.fit(train_X)
    transformed = pipeline.transform(test_X)  # must not raise despite unseen category
    assert transformed.shape[0] == len(test_X)


def test_preprocessing_output_feature_names_are_stable_across_calls(sample_df, project_config):
    target = project_config.data.target_column
    X = sample_df.drop(columns=[target])
    pipeline = build_preprocessing_pipeline(project_config.features, RISK_SCOPE_APPLICATION)
    pipeline.fit(X)
    names_1 = get_output_feature_names(pipeline)
    names_2 = get_output_feature_names(pipeline)
    assert names_1 == names_2
    assert len(names_1) == pipeline.transform(X).shape[1]
