from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.pipelines.batch_inference_pipeline import run_batch_inference_pipeline
from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline


@pytest.fixture
def trained_model(test_settings, test_config):
    return run_training_pipeline(test_config, test_settings)


def test_batch_inference_scores_applications_without_target(
    trained_model, test_settings, test_config, sample_df, tmp_path
):
    applications = sample_df.drop(columns=[test_config.data.target_column]).head(20)
    input_path = tmp_path / "applications.csv"
    applications.to_csv(input_path, index=False)
    output_path = tmp_path / "predictions.csv"

    result = run_batch_inference_pipeline(
        test_config, test_settings, input_path=input_path, output_path=output_path, model_version="latest"
    )

    assert result.output_path.exists()
    predictions = pd.read_csv(result.output_path)
    assert len(predictions) == 20
    assert predictions["default_probability"].between(0, 1).all()
    assert set(predictions["predicted_default"].unique()) <= {0, 1}
    assert result.n_rejected == 0


def test_batch_inference_ignores_target_column_if_present(
    trained_model, test_settings, test_config, sample_df, tmp_path
):
    """The target column must never be required — and if present (e.g. a
    stale export), it must be dropped rather than used or rejected."""
    applications = sample_df.head(10)  # includes default_flag
    input_path = tmp_path / "applications_with_target.csv"
    applications.to_csv(input_path, index=False)
    output_path = tmp_path / "predictions_with_target.csv"

    result = run_batch_inference_pipeline(
        test_config, test_settings, input_path=input_path, output_path=output_path, model_version="latest"
    )
    predictions = pd.read_csv(result.output_path)
    assert len(predictions) == 10
    assert "default_flag" not in predictions.columns


def test_batch_inference_never_refits_the_model(trained_model, test_settings, test_config, sample_df, tmp_path):
    from bnpl_credit_risk.models.persistence import ArtifactBundle
    from bnpl_credit_risk.models.registry import resolve_model_dir

    models_dir = test_settings.resolve(test_settings.artifacts_dir) / "models"
    before = ArtifactBundle.load(resolve_model_dir(models_dir, "latest"))

    applications = sample_df.drop(columns=[test_config.data.target_column]).head(5)
    input_path = tmp_path / "applications.csv"
    applications.to_csv(input_path, index=False)
    run_batch_inference_pipeline(
        test_config, test_settings, input_path=input_path, output_path=tmp_path / "out.csv", model_version="latest"
    )

    after = ArtifactBundle.load(resolve_model_dir(models_dir, "latest"))
    assert before.version == after.version  # no new model version was created by scoring
