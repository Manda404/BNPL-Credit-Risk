"""Environment settings and typed configuration loading.

Two layers are kept deliberately separate:

- `Settings` (pydantic-settings): environment-driven, read from `.env` / real
  env vars. Machine/deployment-specific (paths overrides, log level, MLflow).
- `*Config` models below: read from `configs/*.yaml`. Pipeline-specific,
  versioned in git, meant to be reviewed like code.

Both resolve paths relative to `project_root()`, never absolute paths baked
into source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from bnpl_credit_risk.constants import (
    RISK_SCOPES,
    SPLIT_STRATEGIES,
    THRESHOLD_POLICIES,
    VALIDATION_STRATEGIES,
)
from bnpl_credit_risk.exceptions import ConfigError


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Locate the project root by walking up from this file to find pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise ConfigError("Could not locate project root (no pyproject.toml found above settings.py)")


class Settings(BaseSettings):
    """Environment-driven settings, overridable via `.env` or real env vars (BNPL_ prefix)."""

    model_config = SettingsConfigDict(env_prefix="BNPL_", env_file=".env", extra="ignore")

    data_raw_path: str = "data/raw/BNPL_CreditRisk_Dataset.csv"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    log_level: str = "INFO"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "bnpl-credit-risk"
    random_seed: int = 42

    def resolve(self, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else project_root() / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# configs/base.yaml
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    data_interim: str
    data_processed: str
    data_predictions: str
    data_reports: str


class BaseConfig(BaseModel):
    project_name: str
    random_seed: int
    paths: PathsConfig


# --------------------------------------------------------------------------- #
# configs/data.yaml
# --------------------------------------------------------------------------- #
class ValidationPolicyConfig(BaseModel):
    strategy: Literal["strict", "warn", "quarantine"]
    quarantine_output_dir: str


class BoundConfig(BaseModel):
    min: float
    max: float


class DataConfig(BaseModel):
    validation: ValidationPolicyConfig
    id_column: str
    target_column: str
    date_column: str
    required_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    bounds: dict[str, BoundConfig]
    allowed_values: dict[str, list[str]]


# --------------------------------------------------------------------------- #
# configs/features.yaml
# --------------------------------------------------------------------------- #
class EngineeredFeatureConfig(BaseModel):
    enabled: bool
    requires: list[str]
    leaky: bool
    bins: list[float] | None = None
    labels: list[str] | None = None
    risk_score_threshold: float | None = None


class FeaturesConfig(BaseModel):
    engineered_features: dict[str, EngineeredFeatureConfig]
    leaky_raw_columns: list[str]
    always_excluded_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    behavioral_numeric_features: list[str]
    behavioral_categorical_features: list[str]


# --------------------------------------------------------------------------- #
# configs/model.yaml
# --------------------------------------------------------------------------- #
class CalibrationConfig(BaseModel):
    enabled: bool
    method: Literal["sigmoid", "isotonic"]
    cv: int


class CrossValidationConfig(BaseModel):
    enabled: bool
    n_splits: int
    shuffle: bool


class ModelConfig(BaseModel):
    risk_scope: Literal["application_risk", "behavioral_risk"]
    algorithm: str
    xgboost_params: dict[str, Any]
    calibration: CalibrationConfig
    cross_validation: CrossValidationConfig


# --------------------------------------------------------------------------- #
# configs/training.yaml
# --------------------------------------------------------------------------- #
class SplitConfig(BaseModel):
    strategy: Literal["stratified_random", "group_by_user", "time_based"]
    test_size: float
    random_seed: int


class MLflowRunConfig(BaseModel):
    enabled: bool
    tracking_uri: str
    experiment_name: str


class CostMatrixConfig(BaseModel):
    false_negative_cost: float
    false_positive_cost: float


class ThresholdPolicyConfig(BaseModel):
    policy: Literal["fixed", "best_f1", "min_recall", "min_precision", "cost_matrix"]
    fixed_value: float
    min_recall: float
    min_precision: float
    cost_matrix: CostMatrixConfig


class RiskBandConfig(BaseModel):
    name: str
    max_probability: float


class TrainingConfig(BaseModel):
    split: SplitConfig
    mlflow: MLflowRunConfig
    threshold: ThresholdPolicyConfig
    risk_bands: list[RiskBandConfig]


# --------------------------------------------------------------------------- #
# configs/inference.yaml
# --------------------------------------------------------------------------- #
class OutputConfig(BaseModel):
    format: str
    directory: str
    columns: list[str]


class InferenceConfig(BaseModel):
    model_version: str
    validation: ValidationPolicyConfig
    output: OutputConfig


# --------------------------------------------------------------------------- #
# configs/logging.yaml
# --------------------------------------------------------------------------- #
class ConsoleSinkConfig(BaseModel):
    enabled: bool
    colorize: bool


class FileSinkConfig(BaseModel):
    enabled: bool
    path: str
    rotation: str
    retention: str
    compression: str


class LoggingConfig(BaseModel):
    level: str
    console: ConsoleSinkConfig
    file: FileSinkConfig


# --------------------------------------------------------------------------- #
# Aggregate + loader
# --------------------------------------------------------------------------- #
class ProjectConfig(BaseModel):
    base: BaseConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    logging: LoggingConfig

    model_config = {"protected_namespaces": ()}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping")
    return content


def load_config(
    config_dir: Path | None = None,
    *,
    model_config_path: Path | None = None,
    inference_config_path: Path | None = None,
) -> ProjectConfig:
    """Load and validate the full project configuration.

    Parameters
    ----------
    config_dir:
        Directory containing base.yaml/data.yaml/features.yaml/... Defaults to
        `<project_root>/configs`.
    model_config_path:
        Override for model.yaml — e.g. pass configs/model_behavioral_risk.yaml
        to explicitly opt into the behavioral_risk profile. Defaults to
        `<config_dir>/model.yaml`.
    inference_config_path:
        Override for inference.yaml. Defaults to `<config_dir>/inference.yaml`.
    """
    config_dir = config_dir or (project_root() / "configs")

    base = BaseConfig.model_validate(_load_yaml(config_dir / "base.yaml"))
    data = DataConfig.model_validate(_load_yaml(config_dir / "data.yaml"))
    features = FeaturesConfig.model_validate(_load_yaml(config_dir / "features.yaml"))
    model = ModelConfig.model_validate(
        _load_yaml(model_config_path or (config_dir / "model.yaml"))
    )
    training = TrainingConfig.model_validate(_load_yaml(config_dir / "training.yaml"))
    inference = InferenceConfig.model_validate(
        _load_yaml(inference_config_path or (config_dir / "inference.yaml"))
    )
    logging_cfg = LoggingConfig.model_validate(_load_yaml(config_dir / "logging.yaml"))

    if model.risk_scope not in RISK_SCOPES:
        raise ConfigError(f"Unknown risk_scope '{model.risk_scope}', expected one of {RISK_SCOPES}")
    if training.split.strategy not in SPLIT_STRATEGIES:
        raise ConfigError(f"Unknown split strategy '{training.split.strategy}'")
    if training.threshold.policy not in THRESHOLD_POLICIES:
        raise ConfigError(f"Unknown threshold policy '{training.threshold.policy}'")
    if data.validation.strategy not in VALIDATION_STRATEGIES:
        raise ConfigError(f"Unknown validation strategy '{data.validation.strategy}'")

    return ProjectConfig(
        base=base,
        data=data,
        features=features,
        model=model,
        training=training,
        inference=inference,
        logging=logging_cfg,
    )
