#!/usr/bin/env python3
"""Load -> clean -> validate -> engineer features, and write the result to
data/processed/. Useful to inspect the exact feature set a given risk_scope
produces, or to feed notebooks without re-running validation each time.

Usage: python scripts/prepare_data.py [--output data/processed/bnpl_features.csv]
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.schemas import training_input_schema
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.features.builder import BNPLFeatureBuilder
from bnpl_credit_risk.features.leakage import BNPLLeakageGuard
from bnpl_credit_risk.logging import configure_logging
from bnpl_credit_risk.settings import get_settings, load_config


def main(
    output: Path | None = typer.Option(None, help="Output CSV path. Defaults to data/processed/bnpl_features.csv"),
    config_dir: Path | None = typer.Option(None, help="Override configs/ directory"),
) -> None:
    settings = get_settings()
    config = load_config(config_dir)
    configure_logging(settings, config.logging)
    log = logger.bind(pipeline="scripts.prepare_data")

    df = BNPLDataLoader(settings, config.data).load_raw()
    df = BNPLDataCleaner(config.data).clean(df)

    schema = training_input_schema(config.data)
    validation = BNPLDataValidator(schema, config.data.validation.strategy).validate(df)
    df = validation.valid

    leakage_result = BNPLLeakageGuard(
        config.features,
        config.model.risk_scope,
    ).apply(df)
    df = leakage_result.cleaned

    builder = BNPLFeatureBuilder(
        config.features,
        risk_scope=config.model.risk_scope,
        date_column=config.data.date_column,
    )
    df = builder.transform(df)

    output_path = output or (settings.resolve(config.base.paths.data_processed) / "bnpl_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Prepared features written rows={} columns={} -> {}", df.shape[0], df.shape[1], output_path)


if __name__ == "__main__":
    typer.run(main)
