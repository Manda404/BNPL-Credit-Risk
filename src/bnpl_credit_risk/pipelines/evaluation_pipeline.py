"""Re-evaluate a saved model artifact without retraining.

Rebuilds the identical test split (same strategy, same seed, same source
data) and scores the persisted pipeline against it — useful to regenerate a
report, or to check that a model still performs as expected after a config
or dependency change, without paying for another training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.schemas import training_input_schema
from bnpl_credit_risk.data.splitting import get_splitter
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.evaluation.evaluator import ClassificationEvaluator
from bnpl_credit_risk.evaluation.reports import write_json_report, write_markdown_report
from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.models.registry import resolve_model_dir
from bnpl_credit_risk.settings import ProjectConfig, Settings


def run_evaluation_pipeline(
    config: ProjectConfig, settings: Settings, *, model_version: str = "latest"
) -> dict:
    log = logger.bind(pipeline="pipelines.evaluation")

    models_dir = settings.resolve(settings.artifacts_dir) / "models"
    version_dir = resolve_model_dir(models_dir, model_version)
    bundle = ArtifactBundle.load(version_dir)
    log.info("Loaded model artifact version={}", bundle.version)

    loader = BNPLDataLoader(settings, config.data)
    cleaner = BNPLDataCleaner(config.data)
    df = cleaner.clean(loader.load_raw())

    schema = training_input_schema(config.data)
    validator = BNPLDataValidator(schema, config.data.validation.strategy)
    df = validator.validate(df).valid

    split_config = config.training.split
    splitter = get_splitter(
        split_config.strategy,
        split_config,
        id_column=config.data.id_column,
        date_column=config.data.date_column,
        target_column=config.data.target_column,
    )
    _, test_df = splitter.split(df)
    X_test = test_df.drop(columns=[config.data.target_column])
    y_test = test_df[config.data.target_column].to_numpy()

    y_prob = bundle.pipeline.predict_proba(X_test)[:, 1]
    threshold_grid = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    report = ClassificationEvaluator().evaluate(
        y_test,
        y_prob,
        bundle.threshold["threshold"],
        cost_matrix=config.training.threshold.cost_matrix,
        threshold_grid=threshold_grid,
    )

    reports_dir = settings.resolve(config.base.paths.data_reports)
    write_json_report(report, Path(reports_dir) / f"evaluation_{bundle.version}.json")
    write_markdown_report(
        report, Path(reports_dir) / f"evaluation_{bundle.version}.md", title=f"Evaluation — {bundle.version}"
    )

    log.info("Re-evaluation complete version={} roc_auc={:.4f}", bundle.version, report["roc_auc"])
    return report
