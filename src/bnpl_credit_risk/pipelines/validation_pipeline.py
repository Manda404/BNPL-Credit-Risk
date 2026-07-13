"""Standalone data-validation pipeline: load -> clean -> validate -> quality
report -> (optionally) quarantine files, without training anything. Powers
`bnpl-risk validate` / scripts/validate_data.py — useful to check a new data
drop before committing to a training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from bnpl_credit_risk.constants import VALIDATION_QUARANTINE
from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.quality import DataQualityReport
from bnpl_credit_risk.data.schemas import training_input_schema
from bnpl_credit_risk.data.validation import BNPLDataValidator, ValidationResult
from bnpl_credit_risk.exceptions import DataValidationError
from bnpl_credit_risk.settings import ProjectConfig, Settings


@dataclass
class ValidationPipelineResult:
    validation: ValidationResult
    quality_report: dict
    success: bool
    error: str | None = None


def run_validation_pipeline(
    config: ProjectConfig, settings: Settings, *, input_path: str | Path | None = None
) -> ValidationPipelineResult:
    log = logger.bind(pipeline="pipelines.validation")
    log.info("Starting data validation pipeline")

    loader = BNPLDataLoader(settings, config.data)
    cleaner = BNPLDataCleaner(config.data)
    quality = DataQualityReport(target_column=config.data.target_column)

    df = loader.load_raw(input_path)
    df = cleaner.clean(df)

    report = quality.build(df)
    quality.log(report)

    schema = training_input_schema(config.data)
    validator = BNPLDataValidator(schema, config.data.validation.strategy)

    try:
        result = validator.validate(df)
    except DataValidationError as exc:
        log.error("Validation failed under strict policy: {}", exc)
        return ValidationPipelineResult(
            validation=ValidationResult(valid=df.iloc[0:0], invalid=df),
            quality_report=report,
            success=False,
            error=str(exc),
        )

    if config.data.validation.strategy == VALIDATION_QUARANTINE and len(result.invalid):
        quarantine_dir = settings.resolve(config.data.validation.quarantine_output_dir)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        invalid_path = quarantine_dir / "quarantined_rows.csv"
        result.invalid.to_csv(invalid_path, index=False)
        log.warning("Quarantined {} invalid row(s) -> {}", len(result.invalid), invalid_path)

    log.info("Validation pipeline complete: {} valid rows, {} invalid rows", len(result.valid), len(result.invalid))
    return ValidationPipelineResult(validation=result, quality_report=report, success=True)
