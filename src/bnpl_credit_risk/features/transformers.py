"""Scikit-learn-compatible transformer wrapping BNPLFeatureBuilder.

Wrapping the (stateless) feature engineering step as a proper
`BaseEstimator`/`TransformerMixin` lets it live inside a single `sklearn.Pipeline`
together with the `ColumnTransformer` encoding step and the model — so the
whole thing serializes and deserializes as one artifact, and train/eval/
inference are guaranteed to run the exact same transformation code.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION
from bnpl_credit_risk.features.builder import BNPLFeatureBuilder, ENGINEERED_FEATURE_COLUMNS
from bnpl_credit_risk.settings import FeaturesConfig


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """`fit` is a no-op — every engineered feature is a deterministic function
    of its source columns, so there is no train-only statistic to learn here
    (that happens downstream, in the OneHotEncoder / imputer)."""

    def __init__(
        self,
        features_config: FeaturesConfig,
        risk_scope: str,
        date_column: str = "transaction_date",
    ) -> None:
        self.features_config = features_config
        self.risk_scope = risk_scope
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineeringTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        builder = BNPLFeatureBuilder(
            self.features_config,
            risk_scope=self.risk_scope,
            date_column=self.date_column,
        )
        return builder.transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        base = list(input_features) if input_features is not None else []
        engineered = [
            column
            for column in ENGINEERED_FEATURE_COLUMNS
            if column not in base
            and self.features_config.engineered_features[column].enabled
            and not (
                self.risk_scope == RISK_SCOPE_APPLICATION
                and self.features_config.engineered_features[column].leaky
            )
        ]
        return base + engineered
