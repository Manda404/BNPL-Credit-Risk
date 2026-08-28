"""Materialize deterministic business features for train and test partitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bnpl_credit_risk.data.partitioning import SavedSplitPaths, save_train_test_csv
from bnpl_credit_risk.features.builder import BNPLFeatureBuilder, resolve_feature_columns
from bnpl_credit_risk.settings import ProjectConfig, Settings


@dataclass(frozen=True)
class FeatureEngineeringPipelineResult:
    train: pd.DataFrame
    test: pd.DataFrame
    selected_engineered_features: tuple[str, ...]
    paths: SavedSplitPaths


@dataclass(frozen=True)
class FeatureEngineeringPipeline:
    """Apply the same feature contract to both partitions and persist them."""

    settings: Settings
    config: ProjectConfig

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> FeatureEngineeringPipelineResult:
        builder = BNPLFeatureBuilder(
            features_config=self.config.features,
            risk_scope=self.config.model.risk_scope,
            date_column=self.config.data.date_column,
        )
        train_engineered = builder.transform(train_df)
        test_engineered = builder.transform(test_df)

        if list(train_engineered.columns) != list(test_engineered.columns):
            raise ValueError(
                "Engineered train and test partitions do not have the same columns"
            )

        numeric, categorical = resolve_feature_columns(
            self.config.features,
            self.config.model.risk_scope,
        )
        model_columns = set(numeric) | set(categorical)
        selected = tuple(
            feature_name
            for feature_name, feature_config in self.config.features.engineered_features.items()
            if feature_config.enabled and feature_name in model_columns
        )

        output = self.config.features.feature_output
        paths = save_train_test_csv(
            train_engineered,
            test_engineered,
            output_dir=self.settings.resolve(output.directory),
            train_filename=output.train_filename,
            test_filename=output.test_filename,
        )
        return FeatureEngineeringPipelineResult(
            train=train_engineered,
            test=test_engineered,
            selected_engineered_features=selected,
            paths=paths,
        )
