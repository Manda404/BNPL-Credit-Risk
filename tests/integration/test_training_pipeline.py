from __future__ import annotations

import json

import joblib
import numpy as np

from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline


def test_training_pipeline_end_to_end_produces_artifact(test_settings, test_config):
    result = run_training_pipeline(test_config, test_settings)

    version_dir = test_settings.resolve(test_settings.artifacts_dir) / "models" / result.version
    assert (version_dir / "pipeline.joblib").exists()
    assert (version_dir / "metadata.json").exists()
    assert (version_dir / "metrics.json").exists()
    assert (version_dir / "threshold.json").exists()
    assert (version_dir / "feature_schema.json").exists()

    latest_pointer = json.loads((version_dir.parent / "latest.json").read_text())
    assert latest_pointer["version"] == result.version

    assert 0.0 <= result.test_metrics["roc_auc"] <= 1.0
    assert 0.0 <= result.threshold["threshold"] <= 1.0


def test_saved_pipeline_reproduces_identical_predictions_after_reload(test_settings, test_config):
    result = run_training_pipeline(test_config, test_settings)
    version_dir = test_settings.resolve(test_settings.artifacts_dir) / "models" / result.version

    bundle = ArtifactBundle.load(version_dir)
    raw_pipeline = joblib.load(version_dir / "pipeline.joblib")

    import pandas as pd

    df = pd.read_csv(test_settings.data_raw_path)
    df = df.drop(columns=[test_config.data.target_column])

    proba_from_bundle = bundle.pipeline.predict_proba(df)[:, 1]
    proba_from_raw_load = raw_pipeline.predict_proba(df)[:, 1]
    assert np.allclose(proba_from_bundle, proba_from_raw_load)


def test_feature_schema_matches_configured_scope(test_settings, test_config):
    result = run_training_pipeline(test_config, test_settings)
    version_dir = test_settings.resolve(test_settings.artifacts_dir) / "models" / result.version
    feature_schema = json.loads((version_dir / "feature_schema.json").read_text())
    assert feature_schema["risk_scope"] == test_config.model.risk_scope
    if test_config.model.risk_scope == "application_risk":
        for leaky_col in test_config.features.leaky_raw_columns:
            assert leaky_col not in feature_schema["input_numeric_features"]
            assert leaky_col not in feature_schema["input_categorical_features"]
