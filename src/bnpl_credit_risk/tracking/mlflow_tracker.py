"""Comprehensive MLflow lineage for model comparison and approval."""

from __future__ import annotations

import importlib.metadata
import json
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bnpl_credit_risk.exceptions import ConfigError
from bnpl_credit_risk.models.persistence import current_git_commit
from bnpl_credit_risk.settings import ProjectConfig, project_root
from bnpl_credit_risk.visualization.boosting import BoostingBenchmarkVisualizer
from bnpl_credit_risk.visualization.model_diagnostics import ModelDiagnosticsVisualizer


class BenchmarkMLflowSession(AbstractContextManager["BenchmarkMLflowSession"]):
    def __init__(self, tracker: MLflowTracker, frame: pd.DataFrame, source: Path | None) -> None:
        self.tracker, self.frame, self.source = tracker, frame, source
        self.mlflow = tracker._mlflow()
        self.run_id: str | None = None
        self.dataset: Any = None

    def __enter__(self) -> BenchmarkMLflowSession:
        self.tracker._configure(self.mlflow)
        run = self.mlflow.start_run(
            run_name="boosting-model-comparison",
            tags=self.tracker._tags("model_comparison"),
            description="Same folds, features and metrics for XGBoost, LightGBM and CatBoost.",
            log_system_metrics=self.tracker.config.training.mlflow.log_system_metrics,
        )
        self.run_id = run.info.run_id
        self.dataset = self.tracker._log_dataset(
            self.mlflow, self.frame, "bnpl_development_features", "development", self.source
        )
        self.tracker._log_provenance(self.mlflow)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    @contextmanager
    def model_run(self, name: str) -> Iterator[None]:
        with self.mlflow.start_run(
            run_name=name, nested=True, tags=self.tracker._tags("cross_validation", model_name=name)
        ):
            self.mlflow.log_params(self.tracker._model_params(name))
            self.mlflow.log_params(self.tracker._versions())
            self.mlflow.log_input(self.dataset, context="development")
            yield

    def log_model_result(self, result: Any, y: pd.Series) -> None:
        summary = result.fold_metrics.agg(["mean", "std"])
        for metric in result.fold_metrics.columns:
            key = self.tracker._key(metric)
            self.mlflow.log_metric(f"cv_{key}_mean", float(summary.loc["mean", metric]))
            self.mlflow.log_metric(f"cv_{key}_std", float(summary.loc["std", metric]))
            for fold, value in result.fold_metrics[metric].items():
                self.mlflow.log_metric(f"fold_{key}", float(value), step=int(fold))
        self.mlflow.log_metric("median_best_iteration", result.median_best_iteration)
        self.mlflow.log_table(result.fold_metrics.reset_index(), "tables/fold_metrics.json")
        self.mlflow.log_table(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(y)),
                    "true_label": y.to_numpy(),
                    "oof_probability": result.oof_probability,
                    "oof_prediction": (
                        result.oof_probability
                        >= self.tracker.config.model.benchmark.comparison_threshold
                    ).astype(int),
                }
            ),
            "tables/oof_predictions.json",
        )
        for fold, history in enumerate(result.histories, 1):
            metric = self.tracker._key(history.metric_name)
            for step, (train, valid) in enumerate(
                zip(history.train, history.validation, strict=False), 1
            ):
                self.mlflow.log_metric(f"fold_{fold}_train_{metric}", train, step=step)
                self.mlflow.log_metric(f"fold_{fold}_validation_{metric}", valid, step=step)
        visual = BoostingBenchmarkVisualizer(
            size=self.tracker.config.model.benchmark.visualization.size
        )
        self.mlflow.log_figure(visual.learning_curves(result), "figures/learning_curves.png")
        self.mlflow.log_figure(
            visual.oof_performance(
                y, result, threshold=self.tracker.config.model.benchmark.comparison_threshold
            ),
            "figures/oof_confusion_roc.png",
        )
        self.mlflow.log_figure(visual.metric_summary(result), "figures/cv_metrics.png")

    def complete(self, result: Any) -> None:
        target = self.tracker.config.data.target_column
        self.mlflow.log_params(
            {
                "selected_model": result.best_model_name,
                "selection_metric": self.tracker.config.model.benchmark.selection_metric,
                "n_splits": self.tracker.config.model.cross_validation.n_splits,
                "n_rows": len(self.frame),
                "n_features": len(result.feature_names),
                "target_rate": float(self.frame[target].mean()),
            }
        )
        self.mlflow.log_table(result.comparison.reset_index(), "tables/model_comparison.json")
        visual = BoostingBenchmarkVisualizer(
            size=self.tracker.config.model.benchmark.visualization.size
        )
        self.mlflow.log_figure(visual.comparison(result), "figures/model_comparison.png")
        self.mlflow.set_tag("selected_model", result.best_model_name)
        self.tracker._log_model(
            self.mlflow,
            result.selected_model,
            self.frame[result.feature_names],
            self.tracker.config.model.benchmark.comparison_threshold,
            "candidate_model",
        )


class MLflowTracker:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.training.mlflow.enabled

    def ensure_available(self) -> None:
        if self.enabled:
            self._mlflow()

    def benchmark_session(
        self, frame: pd.DataFrame, source: Path | None
    ) -> BenchmarkMLflowSession | None:
        return BenchmarkMLflowSession(self, frame, source) if self.enabled else None

    def attach_benchmark_artifacts(self, run_id: str | None, path: Path) -> None:
        if not self.enabled or not run_id:
            return
        mlflow = self._mlflow()
        self._configure(mlflow)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(path), artifact_path="benchmark_bundle")
        self._reference(path, run_id, "model_comparison")

    def log_approved_model(
        self,
        *,
        evaluation: Any,
        final_test: Any,
        development_df: pd.DataFrame,
        development_path: Path,
        test_path: Path,
        artifact_dir: Path,
    ) -> tuple[str | None, str | None]:
        if not self.enabled:
            return None, None
        mlflow = self._mlflow()
        self._configure(mlflow)
        name = evaluation.benchmark.best_model_name
        with mlflow.start_run(
            run_name=f"approved-{name}-{artifact_dir.name}",
            tags=self._tags("approved_model", model_name=name, approval_status="approved"),
        ) as run:
            self._log_dataset(
                mlflow, development_df, "bnpl_development_features", "training", development_path
            )
            self._log_dataset(
                mlflow, final_test.frame, "bnpl_reserved_test_features", "testing", test_path
            )
            self._log_provenance(mlflow)
            mlflow.log_params(
                {
                    "model_name": name,
                    "decision_threshold": evaluation.selected_threshold,
                    "threshold_policy": evaluation.threshold_details["policy"],
                    "median_best_iteration": evaluation.benchmark.best_result.median_best_iteration,
                    "n_development_rows": len(development_df),
                    "n_test_rows": len(final_test.frame),
                }
            )
            for metric, value in evaluation.summary.iloc[0].items():
                self._numeric(mlflow, f"oof_{self._key(metric)}", value)
            for metric, value in final_test.metrics.loc["Final test"].items():
                self._numeric(mlflow, f"test_{self._key(metric)}", value)
            self._approved_tables(mlflow, evaluation, final_test)
            self._approved_figures(mlflow, evaluation, final_test)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="approved_bundle")
            info = self._log_model(
                mlflow,
                evaluation.benchmark.selected_model,
                development_df[evaluation.benchmark.feature_names],
                evaluation.selected_threshold,
                "approved_model",
            )
            version = None
            if info is not None:
                registered = mlflow.register_model(
                    info.model_uri,
                    self.config.training.mlflow.registered_model_name,
                    tags={"algorithm": name, "approval_status": "approved"},
                )
                mlflow.MlflowClient().set_registered_model_alias(
                    self.config.training.mlflow.registered_model_name,
                    self.config.training.mlflow.champion_alias,
                    registered.version,
                )
                version = str(registered.version)
                mlflow.set_tag("registered_model_version", version)
            run_id = run.info.run_id
        self._reference(artifact_dir, run_id, "approved_model")
        return run_id, version

    def _approved_tables(self, mlflow: Any, e: Any, t: Any) -> None:
        calibration = e.calibration.copy()
        calibration["Bin"] = calibration["Bin"].astype(str)
        tables = {
            "metrics": t.metrics.reset_index(names="dataset"),
            "calibration": calibration,
            "threshold_tradeoff": e.threshold_tradeoff,
            "lift_and_gains": e.lift_and_gains,
            "feature_importance": e.feature_importance.rename_axis("feature").reset_index(),
            "shap_importance": e.shap.values.abs()
            .mean()
            .rename("mean_abs_shap")
            .sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index(),
            "test_predictions": pd.DataFrame(
                {
                    "true_label": t.y_test,
                    "default_probability": t.probability,
                    "predicted_label": t.prediction,
                }
            ),
        }
        for name, table in tables.items():
            mlflow.log_table(table, f"tables/{name}.json")
        mlflow.log_dict(e.threshold_details, "configuration/threshold.json")

    def _approved_figures(self, mlflow: Any, e: Any, t: Any) -> None:
        name = e.benchmark.best_model_name
        v = ModelDiagnosticsVisualizer(size=self.config.model.benchmark.visualization.size)
        figures = {
            "calibration": v.calibration(e.calibration, e.oof_probability, name),
            "threshold_tradeoff": v.threshold_tradeoff(
                e.threshold_tradeoff, e.selected_threshold, name
            ),
            "lift_and_gains": v.lift_and_gains(e.lift_and_gains, name),
            "feature_importance": v.feature_importance(e.feature_importance, name),
            "shap_global": v.shap_global_bar(e.shap, name),
            "shap_beeswarm": v.shap_beeswarm(e.shap, name),
            "test_confusion_roc": v.classification_performance(
                t.y_test,
                t.probability,
                name,
                threshold=e.selected_threshold,
                dataset_label="Final test",
            ),
        }
        for filename, figure in figures.items():
            mlflow.log_figure(figure, f"figures/{filename}.png")

    def _log_model(
        self, mlflow: Any, model: Any, features: pd.DataFrame, threshold: float, name: str
    ) -> Any:
        if not self.config.training.mlflow.log_model:
            return None
        example = features.head(5).copy()
        signature = self._model_signature(mlflow, example)
        with tempfile.TemporaryDirectory(prefix="bnpl-mlflow-model-") as directory:
            temporary_dir = Path(directory)
            model_path = temporary_dir / "fitted_model.joblib"
            threshold_path = temporary_dir / "threshold.json"
            joblib.dump(model, model_path)
            threshold_path.write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
            return mlflow.pyfunc.log_model(
                name=name,
                python_model=str(
                    project_root() / "src/bnpl_credit_risk/tracking/mlflow_model_code.py"
                ),
                artifacts={
                    "fitted_model": str(model_path),
                    "threshold": str(threshold_path),
                },
                input_example=example,
                signature=signature,
                code_paths=[str(project_root() / "src")],
                pip_requirements=self._requirements(model.model_name),
                metadata={
                    "algorithm": model.model_name,
                    "decision_threshold": threshold,
                    "risk_scope": self.config.model.risk_scope,
                },
            )

    def _log_dataset(
        self, mlflow: Any, frame: pd.DataFrame, name: str, context: str, source: Path | None
    ) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The specified dataset source can be interpreted"
            )
            warnings.filterwarnings(
                "ignore", message="Hint: Inferred schema contains integer column"
            )
            dataset = mlflow.data.from_pandas(
                frame,
                source=source.resolve().as_uri() if source else None,
                targets=self.config.data.target_column,
                name=name,
            )
        mlflow.log_input(
            dataset, context=context, tags={"risk_scope": self.config.model.risk_scope}
        )
        if self.config.training.mlflow.log_dataset_artifacts and source and source.exists():
            mlflow.log_artifact(str(source), artifact_path="datasets")
        return dataset

    def _log_provenance(self, mlflow: Any) -> None:
        mlflow.set_tags(
            {"git_commit": current_git_commit() or "unknown", "mlflow_version": mlflow.__version__}
        )
        mlflow.log_artifacts(str(project_root() / "configs"), artifact_path="configuration/yaml")
        for name in ("pyproject.toml", "poetry.lock"):
            mlflow.log_artifact(str(project_root() / name), artifact_path="environment")

    def _configure(self, mlflow: Any) -> None:
        mlflow.set_tracking_uri(self.config.training.mlflow.tracking_uri)
        name = self.config.training.mlflow.experiment_name
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            eid = mlflow.create_experiment(
                name,
                artifact_location=self.config.training.mlflow.artifact_uri,
                tags={"project": self.config.base.project_name},
            )
            mlflow.set_experiment(experiment_id=eid)
        else:
            mlflow.set_experiment(experiment_id=exp.experiment_id)

    def _tags(self, stage: str, **extra: str) -> dict[str, str]:
        return {
            "project": self.config.base.project_name,
            "environment": "development",
            "stage": stage,
            "risk_scope": self.config.model.risk_scope,
            **extra,
        }

    def _model_params(self, name: str) -> dict[str, Any]:
        if name == "XGBoost":
            return dict(self.config.model.xgboost_params)
        if name == "LightGBM":
            return dict(self.config.model.benchmark.lightgbm_params)
        return dict(self.config.model.benchmark.catboost_params)

    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            f"library_{p}": importlib.metadata.version(p)
            for p in (
                "mlflow",
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "catboost",
            )
        }

    @staticmethod
    def _requirements(model: str) -> list[str]:
        packages = ["mlflow", "pandas", "numpy", "scikit-learn", model.lower()]
        return [f"{p}=={importlib.metadata.version(p)}" for p in packages]

    @staticmethod
    def _model_signature(mlflow: Any, example: pd.DataFrame) -> Any:
        columns = []
        for name, dtype in example.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                data_type = "long"
            elif pd.api.types.is_float_dtype(dtype):
                data_type = "double"
            elif pd.api.types.is_bool_dtype(dtype):
                data_type = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                data_type = "datetime"
            else:
                data_type = "string"
            columns.append(mlflow.types.ColSpec(data_type, name=name, required=False))
        return mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema(columns),
            outputs=mlflow.types.Schema(
                [
                    mlflow.types.ColSpec("double", name="default_probability"),
                    mlflow.types.ColSpec("long", name="predicted_label"),
                ]
            ),
        )

    @staticmethod
    def _key(value: object) -> str:
        return str(value).lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _numeric(mlflow: Any, key: str, value: object) -> None:
        if isinstance(value, int | float | np.integer | np.floating):
            mlflow.log_metric(key, float(value))

    @staticmethod
    def _reference(path: Path, run_id: str, stage: str) -> None:
        (path / "mlflow_run.json").write_text(
            json.dumps({"run_id": run_id, "stage": stage}, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _mlflow() -> Any:
        try:
            import mlflow
        except ImportError as exc:
            raise ConfigError("MLflow is enabled but not installed.") from exc
        return mlflow

    def log_training_pipeline(
        self,
        *,
        metadata: dict[str, Any],
        metrics: dict[str, Any],
        threshold: dict[str, Any],
        artifact_dir: Path,
    ) -> str | None:
        """Keep the historical CLI workflow traceable without registry promotion."""
        if not self.enabled:
            return None
        mlflow = self._mlflow()
        self._configure(mlflow)
        algorithm = str(metadata.get("algorithm", "unknown"))
        with mlflow.start_run(
            run_name=f"legacy-{algorithm}-{artifact_dir.name}",
            tags=self._tags("legacy_training", model_name=algorithm),
            log_system_metrics=self.config.training.mlflow.log_system_metrics,
        ) as run:
            mlflow.log_params(
                {
                    "algorithm": algorithm,
                    "split_strategy": metadata.get("split_strategy", "unknown"),
                    **metadata.get("xgboost_params", {}),
                }
            )
            for section, values in metrics.items():
                if isinstance(values, dict):
                    for name, value in values.items():
                        self._numeric(mlflow, f"{section}_{self._key(name)}", value)
            mlflow.log_dict(threshold, "configuration/threshold.json")
            mlflow.log_artifacts(str(artifact_dir), artifact_path="legacy_bundle")
            self._log_provenance(mlflow)
            run_id = run.info.run_id
        self._reference(artifact_dir, run_id, "legacy_training")
        return run_id
