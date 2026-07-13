"""End-to-end training pipeline.

load config -> load data -> validate -> quality report -> split (train-only
stats from here on) -> select threshold from train out-of-fold predictions ->
fit final pipeline on train -> optional calibration -> evaluate on test ->
persist artifact bundle + figures -> optional MLflow logging.

This is the single place all of those steps are wired together; every step
itself lives in a small, independently testable module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.quality import DataQualityReport
from bnpl_credit_risk.data.schemas import training_input_schema
from bnpl_credit_risk.data.splitting import get_splitter
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.evaluation.evaluator import ClassificationEvaluator
from bnpl_credit_risk.evaluation.thresholding import ThresholdSelector
from bnpl_credit_risk.exceptions import ConfigError, DataValidationError
from bnpl_credit_risk.features.builder import resolve_feature_columns
from bnpl_credit_risk.features.feature_names import get_feature_importances
from bnpl_credit_risk.models.calibration import ProbabilityCalibrator
from bnpl_credit_risk.models.persistence import ArtifactBundle, current_git_commit
from bnpl_credit_risk.models.trainer import ModelTrainer
from bnpl_credit_risk.settings import ProjectConfig, Settings
from bnpl_credit_risk.visualization.calibration import (
    plot_calibration_comparison,
    plot_calibration_curve,
)
from bnpl_credit_risk.visualization.evaluation import (
    plot_confusion_matrix_and_roc,
    plot_feature_importance,
    plot_probability_distribution,
)


@dataclass
class TrainingPipelineResult:
    version_dir: str
    version: str
    test_metrics: dict[str, Any]
    threshold: dict[str, Any]
    cv_summary: dict[str, Any]


def run_training_pipeline(config: ProjectConfig, settings: Settings) -> TrainingPipelineResult:
    log = logger.bind(pipeline="pipelines.training")
    log.info("Starting training pipeline risk_scope={} split={}", config.model.risk_scope, config.training.split.strategy)

    # 1-6: load, clean, validate, quality report
    loader = BNPLDataLoader(settings, config.data)
    cleaner = BNPLDataCleaner(config.data)
    quality = DataQualityReport(target_column=config.data.target_column)

    df = loader.load_raw()
    df = cleaner.clean(df)
    report = quality.build(df)
    quality.log(report)

    schema = training_input_schema(config.data)
    validator = BNPLDataValidator(schema, config.data.validation.strategy)
    try:
        validation_result = validator.validate(df)
    except DataValidationError as exc:
        log.error("Training aborted: data contract violated under strict policy: {}", exc)
        raise

    df = validation_result.valid
    if len(validation_result.invalid):
        log.warning("Excluding {} invalid row(s) from training", len(validation_result.invalid))

    # 7-8: features/target split, train/test split
    target_column = config.data.target_column
    splitter = get_splitter(
        config.training.split.strategy,
        config.training.split,
        id_column=config.data.id_column,
        date_column=config.data.date_column,
        target_column=target_column,
    )
    train_df, test_df = splitter.split(df)
    log.info("Split strategy={} train_rows={} test_rows={}", config.training.split.strategy, len(train_df), len(test_df))

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # 9-11: scale_pos_weight (train only) + threshold selection via out-of-fold CV + final fit
    trainer = ModelTrainer(config.model, config.features)
    scale_pos_weight = trainer.compute_scale_pos_weight(y_train)
    cv_config = config.model.cross_validation

    cv_summary: dict[str, Any] = {}
    threshold_result: dict[str, Any]

    if cv_config.enabled:
        cv = StratifiedKFold(n_splits=cv_config.n_splits, shuffle=cv_config.shuffle, random_state=config.base.random_seed)
        unfit_pipeline = trainer.build_pipeline(scale_pos_weight, date_column=config.data.date_column)
        oof_proba = cross_val_predict(
            unfit_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        cv_auc_scores = cross_val_score(unfit_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_summary = {
            "n_splits": cv_config.n_splits,
            "roc_auc_scores": [float(s) for s in cv_auc_scores],
            "roc_auc_mean": float(np.mean(cv_auc_scores)),
            "roc_auc_std": float(np.std(cv_auc_scores)),
        }
        log.info("Train CV ROC-AUC mean={:.4f} std={:.4f}", cv_summary["roc_auc_mean"], cv_summary["roc_auc_std"])

        threshold_selector = ThresholdSelector(config.training.threshold)
        threshold_result = threshold_selector.select(y_train.to_numpy(), oof_proba)
    else:
        # No CV configured: fall back to the fixed policy value (best-effort, documented limitation).
        threshold_result = {"threshold": config.training.threshold.fixed_value, "policy": "fixed", "rationale": "cross_validation disabled; used fixed_value."}

    threshold_result["risk_bands"] = [rb.model_dump() for rb in config.training.risk_bands]

    pipeline = trainer.train(X_train, y_train, date_column=config.data.date_column)

    if config.model.calibration.enabled:
        calibrator = ProbabilityCalibrator(config.model.calibration)
        uncalibrated_test_proba = pipeline.predict_proba(X_test)[:, 1]
        pipeline = calibrator.calibrate(
            trainer.build_pipeline(scale_pos_weight, date_column=config.data.date_column), X_train, y_train
        )
        calibrated_test_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        uncalibrated_test_proba = None
        calibrated_test_proba = None

    # 12: evaluation on the held-out test split
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    evaluator = ClassificationEvaluator()
    threshold_grid = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    test_metrics = evaluator.evaluate(
        y_test.to_numpy(),
        y_test_proba,
        threshold_result["threshold"],
        cost_matrix=config.training.threshold.cost_matrix,
        threshold_grid=threshold_grid,
    )

    importances = get_feature_importances(pipeline)
    feature_names = list(importances.index)

    numeric_features, categorical_features = resolve_feature_columns(config.features, config.model.risk_scope)

    # 13-19: persist artifact + figures + report
    version = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    figures_dir = settings.resolve(settings.artifacts_dir) / "figures" / version
    plot_confusion_matrix_and_roc(
        y_test.to_numpy(),
        (y_test_proba >= threshold_result["threshold"]).astype(int),
        y_test_proba,
        save_path=figures_dir / "confusion_matrix_and_roc.png",
    )
    plot_probability_distribution(y_test_proba, figures_dir / "probability_distribution.png")
    if len(importances):
        plot_feature_importance(importances, save_path=figures_dir / "feature_importance.png")
    if calibrated_test_proba is not None and uncalibrated_test_proba is not None:
        plot_calibration_comparison(y_test.to_numpy(), uncalibrated_test_proba, calibrated_test_proba, save_path=figures_dir / "calibration_before_after.png")
    else:
        plot_calibration_curve(y_test.to_numpy(), y_test_proba, save_path=figures_dir / "calibration_curve.png")

    metadata = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "risk_scope": config.model.risk_scope,
        "algorithm": config.model.algorithm,
        "xgboost_params": {**config.model.xgboost_params, "scale_pos_weight": scale_pos_weight},
        "calibration": config.model.calibration.model_dump(),
        "split_strategy": config.training.split.strategy,
        "test_size": config.training.split.test_size,
        "random_seed": config.base.random_seed,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "git_commit": current_git_commit(),
        "data_quality_report": report,
    }

    metrics_payload = {"test": test_metrics, "cross_validation_train": cv_summary}

    feature_schema = {
        "schema_version": "1.0",
        "risk_scope": config.model.risk_scope,
        "input_numeric_features": numeric_features,
        "input_categorical_features": categorical_features,
        "output_feature_names": feature_names,
    }

    bundle = ArtifactBundle(
        pipeline=pipeline,
        metadata=metadata,
        metrics=metrics_payload,
        threshold=threshold_result,
        feature_schema=feature_schema,
        version=version,
    )
    models_dir = settings.resolve(settings.artifacts_dir) / "models"
    version_dir = bundle.save(models_dir)

    if config.training.mlflow.enabled:
        _log_to_mlflow(config, metadata, metrics_payload, threshold_result, version_dir)

    log.info(
        "Training pipeline complete version={} test_roc_auc={:.4f} test_brier={:.4f} threshold={:.3f}",
        version,
        test_metrics["roc_auc"],
        test_metrics["brier_score"],
        threshold_result["threshold"],
    )

    return TrainingPipelineResult(
        version_dir=str(version_dir), version=version, test_metrics=test_metrics,
        threshold=threshold_result, cv_summary=cv_summary,
    )


def _log_to_mlflow(config: ProjectConfig, metadata: dict, metrics: dict, threshold: dict, version_dir) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise ConfigError(
            "training.mlflow.enabled=true but the 'mlflow' extra is not installed. "
            "Install it with: poetry install --extras mlflow"
        ) from exc

    mlflow.set_tracking_uri(config.training.mlflow.tracking_uri)
    mlflow.set_experiment(config.training.mlflow.experiment_name)
    with mlflow.start_run(run_name=metadata["version"]):
        mlflow.log_params({"risk_scope": metadata["risk_scope"], **metadata["xgboost_params"]})
        mlflow.log_metrics({k: v for k, v in metrics["test"].items() if isinstance(v, int | float)})
        mlflow.log_dict(threshold, "threshold.json")
        mlflow.log_artifacts(str(version_dir))
