"""Audit and materialize leakage-safe train/test application datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bnpl_credit_risk.data.partitioning import SavedSplitPaths, save_train_test_csv
from bnpl_credit_risk.features.leakage import BNPLLeakageGuard
from bnpl_credit_risk.settings import ProjectConfig, Settings


@dataclass(frozen=True)
class LeakagePipelineResult:
    train: pd.DataFrame
    test: pd.DataFrame
    report: pd.DataFrame
    removed_columns: tuple[str, ...]
    paths: SavedSplitPaths


@dataclass(frozen=True)
class LeakageControlPipeline:
    """Apply the same semantic leakage policy to train and held-out test."""

    settings: Settings
    config: ProjectConfig

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> LeakagePipelineResult:
        guard = BNPLLeakageGuard(self.config.features, self.config.model.risk_scope)
        train_result = guard.apply(train_df)
        test_result = guard.apply(test_df)

        output = self.config.features.leakage_output
        paths = save_train_test_csv(
            train_result.cleaned,
            test_result.cleaned,
            output_dir=self.settings.resolve(output.directory),
            train_filename=output.train_filename,
            test_filename=output.test_filename,
        )
        return LeakagePipelineResult(
            train=train_result.cleaned,
            test=test_result.cleaned,
            report=train_result.report,
            removed_columns=train_result.removed_columns,
            paths=paths,
        )
