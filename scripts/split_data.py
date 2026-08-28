#!/usr/bin/env python3
"""Create reproducible stratified train/test CSV files from the raw dataset."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.logging import configure_logging
from bnpl_credit_risk.pipelines.data_split_pipeline import StratifiedDataSplitPipeline
from bnpl_credit_risk.settings import get_settings, load_config


def main(
    input_path: Path | None = typer.Option(None, help="Input CSV; defaults to BNPL_DATA_RAW_PATH"),
    config_dir: Path | None = typer.Option(None, help="Override configs/ directory"),
) -> None:
    settings = get_settings()
    config = load_config(config_dir)
    configure_logging(settings, config.logging)
    df = BNPLDataLoader(settings, config.data).load_raw(input_path)
    df = BNPLDataCleaner(config.data).clean(df)
    split_pipeline = StratifiedDataSplitPipeline(
        settings=settings,
        config=config,
    )
    result = split_pipeline.run(df)
    logger.bind(pipeline="scripts.split_data").info(
        "Created train_rows={} test_rows={} train={} test={} figure={}",
        len(result.split.train),
        len(result.split.test),
        result.paths.train_path,
        result.paths.test_path,
        result.figure_path,
    )


if __name__ == "__main__":
    typer.run(main)
