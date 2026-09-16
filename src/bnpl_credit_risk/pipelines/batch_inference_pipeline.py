"""Class-based orchestration for configured batch inference and notebook 06."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.inference.batch import BatchInferenceResult, BatchPredictor
from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.settings import ProjectConfig, Settings


@dataclass
class TestSubmissionResult:
    output_path: Path
    model_version: str
    decision_threshold: float
    n_rows: int
    submission: pd.DataFrame
    detailed_predictions: pd.DataFrame

    @property
    def summary(self) -> dict:
        return {
            "model_version": self.model_version,
            "decision_threshold": self.decision_threshold,
            "n_rows": self.n_rows,
            "output_path": str(self.output_path),
            "columns": self.submission.columns.tolist(),
        }


class BatchInferencePipeline:
    """Load an approved artifact and build the labeled test submission."""

    def __init__(self, *, config: ProjectConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings

    @property
    def input_path(self) -> Path:
        return self.settings.resolve(self.config.inference.input_path)

    @property
    def output_path(self) -> Path:
        return (
            self.settings.resolve(self.config.inference.output.directory)
            / self.config.inference.output.filename
        )

    def describe_model(self, model_version: str | None = None) -> dict:
        version = model_version or self.config.inference.model_version
        models_dir = self.settings.resolve(self.settings.artifacts_dir) / "models"
        predictor = Predictor.load(models_dir, version)
        metadata = predictor.bundle.metadata
        approved = bool(metadata.get("approved_for_inference", False))
        ready_for_scoring = approved or not self.config.inference.require_approved_model
        blocking_reason = None
        if not ready_for_scoring:
            blocking_reason = (
                f"Model version '{predictor.bundle.version}' is not approved. "
                "Run notebook 05 through section 8 to publish the validated model."
            )
        return {
            "model_version": predictor.bundle.version,
            "algorithm": metadata.get("algorithm", "unknown"),
            "workflow": metadata.get("workflow", "legacy"),
            "risk_scope": metadata.get("risk_scope", "unknown"),
            "approved_for_inference": approved,
            "ready_for_scoring": ready_for_scoring,
            "blocking_reason": blocking_reason,
            "decision_threshold": predictor.bundle.threshold["threshold"],
            "input_features": predictor.bundle.feature_schema.get("input_features")
            or [
                *predictor.bundle.feature_schema.get("input_numeric_features", []),
                *predictor.bundle.feature_schema.get("input_categorical_features", []),
            ],
            "risk_bands": predictor.bundle.threshold.get("risk_bands", []),
        }

    def load_applications(
        self, input_path: str | Path | None = None
    ) -> pd.DataFrame:
        return BNPLDataLoader(self.settings, self.config.data).load_raw(
            input_path or self.input_path
        )

    def run(
        self,
        *,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        model_version: str | None = None,
    ) -> BatchInferenceResult:
        return BatchPredictor(self.config, self.settings).run(
            input_path or self.input_path,
            output_path or self.output_path,
            model_version=model_version or self.config.inference.model_version,
        )

    def run_test_submission(
        self,
        *,
        model_version: str | None = None,
    ) -> TestSubmissionResult:
        """Scorer le jeu de test puis exporter les résultats détaillés.

        Paramètre :
            model_version : version explicite ; sinon, celle de configs/inference.yaml.

        Retour :
            TestSubmissionResult avec le chemin du fichier, sa version et les tableaux.

        La cible est isolée avant le scoring puis jointe par identifiant, jamais
        par position. L'export conserve la probabilité, la classe et son libellé,
        la vérité terrain, la justesse de la prédiction et la traçabilité du modèle.
        submission_columns définit l'ordre des colonnes sauvegardées. Une colonne
        configurée mais indisponible fait échouer l'export avant son écriture atomique.
        """
        test_df = self.load_applications(self.input_path)
        target = self.config.data.target_column
        identifier = self.config.data.id_column
        if target not in test_df:
            raise KeyError(f"Test dataset is missing true label column: {target}")

        true_labels = test_df.set_index(identifier)[target].astype(int)
        with TemporaryDirectory(prefix="bnpl-submission-") as temporary_directory:
            temporary_output = Path(temporary_directory) / "detailed_predictions.csv"
            batch_result = BatchPredictor(self.config, self.settings).run(
                self.input_path,
                temporary_output,
                model_version=model_version or self.config.inference.model_version,
            )
            detailed = pd.read_csv(batch_result.output_path)

        # Conserver la classe numérique publique et un libellé lisible dans le CSV.
        detailed["default_risk_class"] = detailed["predicted_default"].astype(int)
        detailed["predicted_class"] = detailed["default_risk_class"].map(
            {
                0: "Repayment predicted",
                1: "Non-repayment predicted",
            }
        )
        # Alias historique, pour les consommateurs qui utilisent encore predicted_label.
        detailed["predicted_label"] = detailed["default_risk_class"]
        # Rattacher la vérité uniquement après scoring, même si l'ordre des lignes change.
        detailed["true_label"] = detailed[identifier].map(true_labels).astype(int)
        detailed["prediction_correct"] = detailed["default_risk_class"].eq(detailed["true_label"])
        submission = detailed.loc[:, self.config.inference.submission_columns]
        BatchPredictor.write_atomic(
            submission,
            self.output_path,
            fmt=self.config.inference.output.format,
        )
        return TestSubmissionResult(
            output_path=self.output_path,
            model_version=batch_result.model_version,
            decision_threshold=batch_result.decision_threshold,
            n_rows=len(submission),
            submission=submission,
            detailed_predictions=detailed,
        )

    @staticmethod
    def load_predictions(result: BatchInferenceResult) -> pd.DataFrame:
        if result.output_path.suffix.lower() == ".parquet":
            return pd.read_parquet(result.output_path)
        return pd.read_csv(result.output_path)


def run_batch_inference_pipeline(
    config: ProjectConfig,
    settings: Settings,
    *,
    input_path: str | Path,
    output_path: str | Path,
    model_version: str = "latest",
) -> BatchInferenceResult:
    return BatchPredictor(config, settings).run(input_path, output_path, model_version=model_version)
