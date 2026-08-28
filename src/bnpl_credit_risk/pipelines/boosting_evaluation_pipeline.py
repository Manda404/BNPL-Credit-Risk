"""Post-selection diagnostics for the winner of notebook 04."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.evaluation.diagnostics import (
    ThresholdTradeoffAnalyzer,
    calibration_table,
    evaluation_summary,
    lift_and_gains_table,
)
from bnpl_credit_risk.evaluation.thresholding import ThresholdSelector
from bnpl_credit_risk.models.boosting import (
    BoostingBenchmarkResult,
    ShapExplanation,
    classification_metrics,
)
from bnpl_credit_risk.models.persistence import (
    ArtifactBundle,
    current_git_commit,
    promote_model_version,
)
from bnpl_credit_risk.pipelines.boosting_benchmark_pipeline import (
    BoostingBenchmarkPipeline,
)
from bnpl_credit_risk.settings import ProjectConfig, Settings
from bnpl_credit_risk.tracking.mlflow_tracker import MLflowTracker


@dataclass
class BoostingEvaluationResult:
    benchmark: BoostingBenchmarkResult
    selected_threshold: float
    threshold_details: dict
    summary: pd.DataFrame
    calibration: pd.DataFrame
    threshold_tradeoff: pd.DataFrame
    lift_and_gains: pd.DataFrame
    feature_importance: pd.Series
    shap: ShapExplanation
    y_development: np.ndarray
    oof_probability: np.ndarray


@dataclass
class FinalTestEvaluation:
    metrics: pd.DataFrame
    y_test: np.ndarray
    probability: np.ndarray
    prediction: np.ndarray
    frame: pd.DataFrame


class BoostingEvaluationPipeline:
    """Load the selected benchmark and produce leakage-safe OOF diagnostics."""

    def __init__(self, *, settings: Settings, config: ProjectConfig) -> None:
        self.settings = settings
        self.config = config
        self.benchmark_pipeline = BoostingBenchmarkPipeline(settings=settings, config=config)
        self.tracker = MLflowTracker(config)
        self.tracker.ensure_available()
        self.last_mlflow_run_id: str | None = None
        self.last_mlflow_registry_version: str | None = None

    def load_benchmark(self, version: str = "latest") -> BoostingBenchmarkResult:
        return self.benchmark_pipeline.load(version)

    def load_development(self) -> pd.DataFrame:
        return self.benchmark_pipeline.load_development()

    def analyze(
        self,
        benchmark: BoostingBenchmarkResult,
        development_df: pd.DataFrame,
    ) -> BoostingEvaluationResult:
        target = self.config.data.target_column
        y_development = development_df[target].astype(int).to_numpy()
        probability = benchmark.best_result.oof_probability
        if len(y_development) != len(probability):
            raise ValueError(
                "Development rows no longer match saved OOF predictions. "
                "Rerun notebook 04 before notebook 05."
            )

        threshold_details = ThresholdSelector(self.config.training.threshold).select(
            y_development, probability
        )
        selected_threshold = float(threshold_details["threshold"])
        benchmark_config = self.config.model.benchmark
        candidates = np.arange(
            benchmark_config.threshold_grid_start,
            benchmark_config.threshold_grid_stop,
            benchmark_config.threshold_grid_step,
        )
        calibration = calibration_table(y_development, probability)
        tradeoff = ThresholdTradeoffAnalyzer(self.config.training.threshold.cost_matrix).build(
            y_development,
            probability,
            candidates,
            selected_threshold,
        )
        lift = lift_and_gains_table(y_development, probability)

        feature_frame = development_df.loc[:, benchmark.feature_names]
        sample_size = min(
            benchmark_config.visualization.shap_sample_size,
            len(feature_frame),
        )
        shap_sample = feature_frame.sample(
            n=sample_size,
            random_state=self.config.base.random_seed,
        )
        explanation = benchmark.selected_model.shap_values(shap_sample)
        summary = evaluation_summary(
            y_development,
            probability,
            selected_threshold,
            calibration,
        )
        return BoostingEvaluationResult(
            benchmark=benchmark,
            selected_threshold=selected_threshold,
            threshold_details=threshold_details,
            summary=summary,
            calibration=calibration,
            threshold_tradeoff=tradeoff,
            lift_and_gains=lift,
            feature_importance=benchmark.selected_model.feature_importances(),
            shap=explanation,
            y_development=y_development,
            oof_probability=probability,
        )

    def evaluate_test(
        self,
        evaluation: BoostingEvaluationResult,
    ) -> FinalTestEvaluation:
        """Open the reserved test once, after model and threshold are frozen."""
        feature_directory = self.settings.resolve(self.config.features.feature_output.directory)
        test_path = feature_directory / self.config.features.feature_output.test_filename
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test features: {test_path}")
        test_df = BNPLDataLoader(self.settings, self.config.data).load_raw(test_path)
        target = self.config.data.target_column
        missing = [
            feature for feature in evaluation.benchmark.feature_names if feature not in test_df
        ]
        if missing:
            raise KeyError(f"Test dataset is missing model features: {missing}")
        y_test = test_df[target].astype(int).to_numpy()
        probability = evaluation.benchmark.selected_model.predict_default_probability(test_df)
        prediction = (probability >= evaluation.selected_threshold).astype(int)
        test_metrics = classification_metrics(
            y_test,
            probability,
            evaluation.selected_threshold,
        )
        comparison = pd.concat(
            [
                evaluation.summary.drop(columns=["Expected calibration error"]),
                pd.DataFrame([test_metrics], index=["Final test"]),
            ]
        )
        return FinalTestEvaluation(
            metrics=comparison,
            y_test=y_test,
            probability=probability,
            prediction=prediction,
            frame=test_df,
        )

    def publish(
        self,
        evaluation: BoostingEvaluationResult,
        final_test: FinalTestEvaluation,
    ) -> Path:
        """Publish an approved model bundle only after final test evaluation."""
        version = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        benchmark = evaluation.benchmark
        threshold = {
            **evaluation.threshold_details,
            "threshold": evaluation.selected_threshold,
            "risk_bands": [band.model_dump() for band in self.config.training.risk_bands],
        }
        metadata = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "approved_for_inference": True,
            "workflow": "boosting_benchmark_04_05",
            "risk_scope": self.config.model.risk_scope,
            "algorithm": benchmark.best_model_name,
            "selection_metric": self.config.model.benchmark.selection_metric,
            "median_best_iteration": benchmark.best_result.median_best_iteration,
            "n_development_rows": len(evaluation.y_development),
            "n_test_rows": len(final_test.y_test),
            "git_commit": current_git_commit(),
        }
        metrics = {
            "development_oof": evaluation.summary.iloc[0].to_dict(),
            "final_test": final_test.metrics.loc["Final test"].to_dict(),
        }
        feature_schema = {
            "schema_version": "1.0",
            "risk_scope": self.config.model.risk_scope,
            "input_numeric_features": benchmark.numeric_features,
            "input_categorical_features": benchmark.categorical_features,
            "input_features": benchmark.feature_names,
        }
        model_card = (
            f"# BNPL Credit Risk — {benchmark.best_model_name}\n\n"
            "Approved after out-of-fold model selection, calibration diagnostics, "
            "threshold analysis and one final test evaluation in notebooks 04/05.\n"
        )
        bundle = ArtifactBundle(
            pipeline=benchmark.selected_model,
            metadata=metadata,
            metrics=metrics,
            threshold=threshold,
            feature_schema=feature_schema,
            model_card=model_card,
            version=version,
        )
        models_dir = self.settings.resolve(self.settings.artifacts_dir) / "models"
        version_dir = bundle.save(models_dir, set_as_latest=False)
        development_path = (
            self.settings.resolve(self.config.features.feature_output.directory)
            / self.config.features.feature_output.train_filename
        )
        test_path = (
            self.settings.resolve(self.config.features.feature_output.directory)
            / self.config.features.feature_output.test_filename
        )
        development_df = self.load_development()
        (
            self.last_mlflow_run_id,
            self.last_mlflow_registry_version,
        ) = self.tracker.log_approved_model(
            evaluation=evaluation,
            final_test=final_test,
            development_df=development_df,
            development_path=development_path,
            test_path=test_path,
            artifact_dir=version_dir,
        )
        promote_model_version(models_dir, version)
        return version_dir
