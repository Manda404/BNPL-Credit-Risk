"""Vérifie le CSV enrichi du notebook 06 avec des artefacts isolés et déterministes."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.pipelines.batch_inference_pipeline import BatchInferencePipeline


@pytest.fixture
def submission_setup(sample_df, test_config, test_settings, tmp_path):
    """Créer un jeu étiqueté et diriger tous les fichiers générés vers tmp_path."""
    source = tmp_path / "test.csv"
    sample_df.head(8).sample(frac=1, random_state=42).to_csv(source, index=False)
    inference = test_config.inference.model_copy(update={
        "input_path": str(source),
        "output": test_config.inference.output.model_copy(update={"directory": str(tmp_path)}),
    })
    config = test_config.model_copy(update={"inference": inference})
    return config, test_settings, source


@pytest.mark.parametrize("threshold", [0.4, 0.6])
def test_submission_persists_scores_and_reconciles_labels(submission_setup, monkeypatch, threshold):
    """Contrôler les deux classes et l'alignement par ID malgré des scores réordonnés."""
    config, settings, source = submission_setup
    bundle = ArtifactBundle(
        pipeline=DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1]),
        metadata={"approved_for_inference": True}, metrics={}, feature_schema={},
        threshold={"threshold": threshold, "risk_bands": [
            {"name": "Low Risk", "max_probability": 0.25},
            {"name": "High Risk", "max_probability": 1.01},
        ]},
        version="submission-test",
    )
    bundle.save(settings.resolve(settings.artifacts_dir) / "models")
    original_score = Predictor.score

    def reversed_scores(self, frame, id_column):
        """Vérifier l'absence de cible et inverser les scores pour tester la jointure."""
        assert config.data.target_column not in frame.columns
        return original_score(self, frame, id_column).iloc[::-1].reset_index(drop=True)

    monkeypatch.setattr(Predictor, "score", reversed_scores)
    result = BatchInferencePipeline(config=config, settings=settings).run_test_submission()
    saved = pd.read_csv(result.output_path)
    source_frame = pd.read_csv(source)
    expected_labels = source_frame.set_index("user_id")[config.data.target_column]

    assert saved.columns.tolist() == config.inference.submission_columns
    assert len(saved) == len(source_frame)
    assert saved["user_id"].tolist() == source_frame["user_id"].tolist()[::-1]
    assert saved["true_label"].tolist() == saved["user_id"].map(expected_labels).tolist()
    assert saved["default_probability"].eq(0.5).all()
    expected_class = int(threshold <= 0.5)
    assert saved["default_risk_class"].eq(expected_class).all()
    assert saved["predicted_label"].equals(saved["default_risk_class"])
    expected_text = {
        0: "Repayment predicted",
        1: "Non-repayment predicted",
    }[expected_class]
    assert saved["predicted_class"].eq(expected_text).all()
    assert saved["prediction_correct"].equals(saved["default_risk_class"].eq(saved["true_label"]))
    assert saved["decision_threshold"].eq(threshold).all()
    assert saved["risk_band"].eq("High Risk").all()
    assert saved["model_version"].eq("submission-test").all()
    assert pd.to_datetime(saved["scoring_timestamp"]).notna().all()
    assert config.data.target_column not in saved
    pd.testing.assert_frame_equal(saved, result.submission)


def test_missing_target_does_not_overwrite_submission(submission_setup):
    """Sans vérité terrain, ne pas produire un fichier d'évaluation trompeur."""
    config, settings, source = submission_setup
    frame = pd.read_csv(source).drop(columns=[config.data.target_column])
    frame.to_csv(source, index=False)
    destination = Path(config.inference.output.directory) / config.inference.output.filename
    destination.write_text("previous-result\n", encoding="utf-8")
    with pytest.raises(KeyError, match="true label"):
        BatchInferencePipeline(config=config, settings=settings).run_test_submission()
    assert destination.read_text(encoding="utf-8") == "previous-result\n"
