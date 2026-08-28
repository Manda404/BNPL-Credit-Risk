"""Validate, stratify, visualize and persist an in-memory dataframe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from bnpl_credit_risk.data.partitioning import (
    DatasetSplit,
    SavedSplitPaths,
    save_train_test_csv,
    stratified_train_test_split,
)
from bnpl_credit_risk.data.schemas import (
    prepared_training_input_schema,
    training_input_schema,
)
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.settings import ProjectConfig, Settings
from bnpl_credit_risk.visualization.splitting import plot_split_target_distribution


@dataclass(frozen=True)
class DataSplitPipelineResult:
    split: DatasetSplit
    paths: SavedSplitPaths
    figure: Figure
    figure_path: Path


@dataclass(frozen=True)
class StratifiedDataSplitPipeline:
    """Orchestrate validation, stratified splitting, persistence and plotting."""

    settings: Settings
    config: ProjectConfig

    def run(
        self,
        df: pd.DataFrame,
        *,
        input_stage: Literal["raw", "prepared"] = "raw",
    ) -> DataSplitPipelineResult:
        """Split ``df`` while preserving its target distribution.

        The dataframe is an explicit input so notebooks and callers always
        know which in-memory dataset is being split. Loading and cleaning
        remain the responsibility of the caller.
        """
        log = logger.bind(pipeline="pipelines.data_split")
        split_config = self.config.data_split
        log.info(
            "Starting stratified data split pipeline test_size={:.1%}",
            split_config.test_size,
        )

        if input_stage == "prepared":
            schema = prepared_training_input_schema(
                self.config.data,
                self.config.features,
                self.config.model.risk_scope,
            )
        else:
            schema = training_input_schema(self.config.data)
        validation = BNPLDataValidator(
            schema, self.config.data.validation.strategy
        ).validate(df)
        dataframe = validation.valid

        split = stratified_train_test_split(
            dataframe,
            target_column=self.config.data.target_column,
            test_size=split_config.test_size,
            random_seed=split_config.random_seed,
        )
        resolved_output_dir = self.settings.resolve(split_config.output.directory)
        paths = save_train_test_csv(
            split.train,
            split.test,
            output_dir=resolved_output_dir,
            train_filename=split_config.output.train_filename,
            test_filename=split_config.output.test_filename,
        )

        resolved_figure_path = self.settings.resolve(split_config.visualization.path)
        figure = plot_split_target_distribution(
            split.train,
            split.test,
            target_column=split.target_column,
            size=split_config.visualization.size,
            save_path=resolved_figure_path,
        )
        log.info(
            "Data split pipeline complete train={} test={} figure={}",
            paths.train_path,
            paths.test_path,
            resolved_figure_path,
        )
        return DataSplitPipelineResult(
            split=split,
            paths=paths,
            figure=figure,
            figure_path=resolved_figure_path,
        )
