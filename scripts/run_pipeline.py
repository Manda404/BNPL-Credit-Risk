#!/usr/bin/env python3
"""Run validate -> train -> evaluate in sequence, stopping at the first
failure. This is what a scheduled "retrain" job should call.

Usage: python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys

from loguru import logger

from bnpl_credit_risk.exceptions import BNPLError, DataValidationError
from bnpl_credit_risk.logging import configure_logging
from bnpl_credit_risk.pipelines.evaluation_pipeline import run_evaluation_pipeline
from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline
from bnpl_credit_risk.pipelines.validation_pipeline import run_validation_pipeline
from bnpl_credit_risk.settings import get_settings, load_config


def main() -> int:
    settings = get_settings()
    config = load_config()
    configure_logging(settings, config.logging)
    log = logger.bind(pipeline="scripts.run_pipeline")

    try:
        validation = run_validation_pipeline(config, settings)
        if not validation.success:
            log.error("Pipeline stopped: validation failed — {}", validation.error)
            return 1

        training = run_training_pipeline(config, settings)
        log.info("Training complete version={}", training.version)

        run_evaluation_pipeline(config, settings, model_version=training.version)
    except DataValidationError as exc:
        log.error("Pipeline stopped: {}", exc)
        return 1
    except BNPLError:
        log.exception("Pipeline stopped: technical error")
        return 2

    log.info("run_pipeline complete: validate -> train -> evaluate all succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
