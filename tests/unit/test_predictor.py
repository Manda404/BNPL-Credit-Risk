from __future__ import annotations

from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.models.trainer import ModelTrainer


def _fit_tiny_pipeline(sample_df, project_config):
    target = project_config.data.target_column
    X = sample_df.drop(columns=[target])
    y = sample_df[target]
    model_config = project_config.model.model_copy(
        update={"xgboost_params": {**project_config.model.xgboost_params, "n_estimators": 10, "max_depth": 2}}
    )
    trainer = ModelTrainer(model_config, project_config.features)
    return trainer.train(X, y, date_column=project_config.data.date_column)


def test_predictor_score_output_schema(sample_df, project_config):
    pipeline = _fit_tiny_pipeline(sample_df, project_config)
    bundle = ArtifactBundle(
        pipeline=pipeline,
        metadata={},
        metrics={},
        threshold={
            "threshold": 0.3,
            "policy": "fixed",
            "risk_bands": [
                {"name": "Low Risk", "max_probability": 0.25},
                {"name": "Medium Risk", "max_probability": 0.5},
                {"name": "High Risk", "max_probability": 0.75},
                {"name": "Very High Risk", "max_probability": 1.01},
            ],
        },
        feature_schema={},
        version="test-version",
    )
    predictor = Predictor(bundle)
    scored = predictor.score(sample_df.drop(columns=[project_config.data.target_column]), "user_id")

    expected_columns = {
        "user_id",
        "default_probability",
        "predicted_default",
        "decision_threshold",
        "risk_band",
        "model_version",
        "scoring_timestamp",
    }
    assert set(scored.columns) == expected_columns
    assert len(scored) == len(sample_df)
    assert scored["default_probability"].between(0, 1).all()
    assert set(scored["predicted_default"].unique()) <= {0, 1}
    assert (scored["model_version"] == "test-version").all()
    assert (scored["decision_threshold"] == 0.3).all()
