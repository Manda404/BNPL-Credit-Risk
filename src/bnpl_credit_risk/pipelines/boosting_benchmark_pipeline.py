"""Pipeline for the leakage-safe three-model benchmark in notebook 04."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.features.builder import resolve_feature_columns
from bnpl_credit_risk.models.benchmark_persistence import BoostingBenchmarkStore
from bnpl_credit_risk.models.boosting import BoostingBenchmark, BoostingBenchmarkResult
from bnpl_credit_risk.settings import ProjectConfig, Settings
from bnpl_credit_risk.tracking.mlflow_tracker import MLflowTracker


@dataclass
class BenchmarkDataset:
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    excluded_engineered_features: list[str]
    source_path: Path | None = None


class BoostingBenchmarkPipeline:
    """Prepare development features, compare models and persist the winner."""

    def __init__(self, *, settings: Settings, config: ProjectConfig) -> None:
        self.settings = settings
        self.config = config
        benchmark_root = (
            settings.resolve(settings.artifacts_dir) / config.model.benchmark.artifact_subdirectory
        )
        self.store = BoostingBenchmarkStore(benchmark_root)
        self.tracker = MLflowTracker(config)
        self.tracker.ensure_available()
        self.last_mlflow_run_id: str | None = None
        self._development_source_path: Path | None = None

    def load_development(self) -> pd.DataFrame:
        feature_directory = self.settings.resolve(self.config.features.feature_output.directory)
        path = feature_directory / self.config.features.feature_output.train_filename
        if not path.exists():
            raise FileNotFoundError(f"Missing development features: {path}. Run notebook 03 first.")
        self._development_source_path = path
        return BNPLDataLoader(self.settings, self.config.data).load_raw(path)

    def prepare(self, df: pd.DataFrame) -> BenchmarkDataset:
        target = self.config.data.target_column
        if target not in df:
            raise KeyError(f"Development dataset is missing target column: {target}")

        configured_numeric, configured_categorical = resolve_feature_columns(
            self.config.features,
            self.config.model.risk_scope,
        )
        engineered = set(self.config.features.engineered_features)
        numeric = [name for name in configured_numeric if name not in engineered]
        categorical = [name for name in configured_categorical if name not in engineered]
        feature_names = [*numeric, *categorical]
        missing = [name for name in feature_names if name not in df]
        if missing:
            raise KeyError(f"Development dataset is missing model features: {missing}")

        return BenchmarkDataset(
            frame=df.copy(),
            X=df.loc[:, feature_names].reset_index(drop=True),
            y=df[target].astype(int).reset_index(drop=True),
            numeric_features=numeric,
            categorical_features=categorical,
            excluded_engineered_features=sorted(engineered.intersection(df.columns)),
        )

    def run(self, df: pd.DataFrame) -> BoostingBenchmarkResult:
        dataset = self.prepare(df)
        benchmark = BoostingBenchmark(
            self.config.model,
            numeric_features=dataset.numeric_features,
            categorical_features=dataset.categorical_features,
            random_seed=self.config.base.random_seed,
        )
        session = self.tracker.benchmark_session(dataset.frame, self._development_source_path)
        if session is None:
            return benchmark.run(dataset.X, dataset.y)
        with session:
            result = benchmark.run(dataset.X, dataset.y, observer=session)
        self.last_mlflow_run_id = session.run_id
        return result

    def save(self, result: BoostingBenchmarkResult) -> Path:
        artifact_dir = self.store.save(result)
        self.tracker.attach_benchmark_artifacts(self.last_mlflow_run_id, artifact_dir)
        return artifact_dir

    def load(self, version: str = "latest") -> BoostingBenchmarkResult:
        return self.store.load(version)
