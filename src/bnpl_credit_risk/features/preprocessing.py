"""Single, serializable preprocessing pipeline.

Replaces the notebook's per-column `LabelEncoder` loop (fit on the full
dataframe, never persisted, no unknown-category handling) with a
`ColumnTransformer` fit on the train split only, wrapped together with
feature engineering in one `sklearn.Pipeline` so the exact same object is
reused for training, evaluation and inference.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bnpl_credit_risk.features.builder import resolve_feature_columns
from bnpl_credit_risk.features.transformers import FeatureEngineeringTransformer
from bnpl_credit_risk.settings import FeaturesConfig


def build_preprocessing_pipeline(
    features_config: FeaturesConfig, risk_scope: str, date_column: str = "transaction_date"
) -> Pipeline:
    numeric_features, categorical_features = resolve_feature_columns(features_config, risk_scope)

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            (
                "feature_engineering",
                FeatureEngineeringTransformer(
                    features_config,
                    risk_scope=risk_scope,
                    date_column=date_column,
                ),
            ),
            ("column_transform", column_transformer),
        ]
    )
