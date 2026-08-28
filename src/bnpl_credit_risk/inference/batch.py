"""BatchPredictor — file-in, file-out batch scoring.

Intended to run unattended (cron / scheduled CI / Airflow later) against a
CSV of applications to score. Never fits or refits anything — it only loads
an already-trained artifact and calls `.predict_proba`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.schemas import inference_input_schema
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.exceptions import DataValidationError, InferenceError
from bnpl_credit_risk.features.leakage import BNPLLeakageGuard
from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.settings import ProjectConfig, Settings


@dataclass
class BatchInferenceResult:
    output_path: Path
    model_version: str
    decision_threshold: float
    n_input: int
    n_scored: int
    n_rejected: int
    rejected_report_path: Path | None
    risk_band_distribution: dict[str, int]

    @property
    def summary(self) -> dict:
        return {
            "model_version": self.model_version,
            "decision_threshold": self.decision_threshold,
            "n_input": self.n_input,
            "n_scored": self.n_scored,
            "n_rejected": self.n_rejected,
            "output_path": str(self.output_path),
            "rejected_report_path": (
                str(self.rejected_report_path) if self.rejected_report_path else None
            ),
        }


class BatchPredictor:
    def __init__(self, config: ProjectConfig, settings: Settings) -> None:
        self._config = config
        self._settings = settings

    def run(
        self, input_path: str | Path, output_path: str | Path, *, model_version: str = "latest"
    ) -> BatchInferenceResult:
        log = logger.bind(pipeline="inference.batch")
        config, settings = self._config, self._settings

        models_dir = settings.resolve(settings.artifacts_dir) / "models"
        predictor = Predictor.load(models_dir, model_version)
        log.info("Loaded model version={} for batch scoring", predictor.bundle.version)
        if config.inference.require_approved_model and not predictor.bundle.metadata.get(
            "approved_for_inference", False
        ):
            raise InferenceError(
                f"Model version '{predictor.bundle.version}' is not approved for inference. "
                "Complete notebook 05 and publish the approved model first."
            )

        loader = BNPLDataLoader(settings, config.data)
        cleaner = BNPLDataCleaner(config.data)
        df = loader.load_raw(input_path)
        n_input = len(df)

        target_column = config.data.target_column
        if target_column in df.columns:
            log.warning("Input contains target column '{}' — dropped before scoring (not used)", target_column)
            df = df.drop(columns=[target_column])

        df = cleaner.clean(df)

        leakage_result = BNPLLeakageGuard(
            config.features,
            config.model.risk_scope,
        ).apply(df)
        df = leakage_result.cleaned
        if leakage_result.removed_columns:
            log.warning(
                "Removed leakage columns before inference: {}",
                leakage_result.removed_columns,
            )

        schema = inference_input_schema(
            config.data,
            config.features,
            config.model.risk_scope,
        )
        strategy = config.inference.validation.strategy
        validator = BNPLDataValidator(schema, strategy)
        try:
            validation_result = validator.validate(df)
        except DataValidationError as exc:
            log.error("Batch inference aborted: {}", exc)
            raise

        scoreable_df = validation_result.valid
        rejected_df = validation_result.invalid

        artifact_features = predictor.bundle.feature_schema.get("input_features", [])
        missing_artifact_features = [
            feature for feature in artifact_features if feature not in scoreable_df
        ]
        if missing_artifact_features:
            raise InferenceError(
                "Input does not satisfy the published model feature contract: "
                f"missing {missing_artifact_features}"
            )

        rejected_report_path = None
        if len(rejected_df):
            reports_dir = settings.resolve(config.inference.validation.quarantine_output_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            rejected_report_path = reports_dir / f"rejected_rows_{Path(output_path).stem}.csv"
            rejected_df.to_csv(rejected_report_path, index=False)
            log.warning("Rejected {} row(s) failed validation -> {}", len(rejected_df), rejected_report_path)

        if scoreable_df.empty:
            raise InferenceError("No valid rows to score after validation")

        predictions = predictor.score(scoreable_df, config.data.id_column)
        output_columns = config.inference.output.columns
        missing_output_columns = [
            column for column in output_columns if column not in predictions
        ]
        if missing_output_columns:
            raise InferenceError(
                f"Configured inference output columns are unavailable: {missing_output_columns}"
            )
        predictions = predictions.loc[:, output_columns]

        output_path = Path(output_path)
        self.write_atomic(predictions, output_path, fmt=config.inference.output.format)

        risk_band_distribution: dict[str, int] = {
            str(k): int(v) for k, v in predictions["risk_band"].value_counts().items()
        }
        log.info(
            "Batch inference complete: scored={} rejected={} output={} risk_bands={}",
            len(predictions),
            len(rejected_df),
            output_path,
            risk_band_distribution,
        )

        return BatchInferenceResult(
            output_path=output_path,
            model_version=predictor.bundle.version,
            decision_threshold=float(predictor.bundle.threshold["threshold"]),
            n_input=n_input,
            n_scored=len(predictions),
            n_rejected=len(rejected_df),
            rejected_report_path=rejected_report_path,
            risk_band_distribution=risk_band_distribution,
        )

    @staticmethod
    def write_atomic(df: pd.DataFrame, output_path: Path, *, fmt: str = "csv") -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
        os.close(fd)
        try:
            if fmt == "csv":
                df.to_csv(tmp_path, index=False)
            elif fmt == "parquet":
                df.to_parquet(tmp_path, index=False)
            else:
                raise InferenceError(f"Unsupported output format: {fmt}")
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
