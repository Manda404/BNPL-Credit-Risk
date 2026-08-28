"""Data contract: what a training dataframe vs an inference dataframe must look like.

Two distinct schemas exist because they have genuinely different requirements —
merging them into one would let the target column silently become "optional but
tolerated" everywhere, which is exactly how target leakage sneaks into
production. See docs/data_contract.md.
"""

from __future__ import annotations

from pydantic import BaseModel

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION
from bnpl_credit_risk.settings import BoundConfig, DataConfig, FeaturesConfig


class DataSchema(BaseModel):
    """Column-level contract used by BNPLDataValidator."""

    id_column: str
    date_column: str
    target_column: str | None
    """None means the target is not expected — and must not be relied on — for this schema."""

    required_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    bounds: dict[str, BoundConfig]
    allowed_values: dict[str, list[str]]


def training_input_schema(data_config: DataConfig) -> DataSchema:
    """Schema for training data: the target column is required and validated as binary."""
    return DataSchema(
        id_column=data_config.id_column,
        date_column=data_config.date_column,
        target_column=data_config.target_column,
        required_columns=data_config.required_columns,
        numeric_columns=data_config.numeric_columns,
        categorical_columns=data_config.categorical_columns,
        bounds=data_config.bounds,
        allowed_values=data_config.allowed_values,
    )


def prepared_training_input_schema(
    data_config: DataConfig,
    features_config: FeaturesConfig,
    risk_scope: str,
) -> DataSchema:
    """Schema for a feature-engineered dataframe that still contains its target.

    Application-risk datasets have already passed through the leakage guard, so
    post-origination columns must not be required here. Conversely, the model
    features selected for the active risk scope are required: this prevents a
    prepared file with a silently missing feature from reaching training.
    """
    numeric_features = list(features_config.numeric_features)
    categorical_features = list(features_config.categorical_features)
    if risk_scope != RISK_SCOPE_APPLICATION:
        numeric_features += list(features_config.behavioral_numeric_features)
        categorical_features += list(features_config.behavioral_categorical_features)

    required_columns = list(
        dict.fromkeys(
            [
                data_config.id_column,
                data_config.date_column,
                data_config.target_column,
                *numeric_features,
                *categorical_features,
            ]
        )
    )
    return DataSchema(
        id_column=data_config.id_column,
        date_column=data_config.date_column,
        target_column=data_config.target_column,
        required_columns=required_columns,
        numeric_columns=numeric_features,
        categorical_columns=categorical_features,
        bounds={
            column: bound
            for column, bound in data_config.bounds.items()
            if column in required_columns
        },
        allowed_values={
            column: values
            for column, values in data_config.allowed_values.items()
            if column in required_columns
        },
    )


def inference_input_schema(
    data_config: DataConfig,
    features_config: FeaturesConfig | None = None,
    risk_scope: str | None = None,
) -> DataSchema:
    """Schema for inference data: the target column is neither required nor used.

    If present in the input file it is dropped before scoring — batch inference
    input is typically "applications to score", which by definition have no
    known outcome yet.
    """
    excluded = {data_config.target_column}
    if features_config is not None and risk_scope == RISK_SCOPE_APPLICATION:
        excluded.update(features_config.leaky_raw_columns)

    required = [column for column in data_config.required_columns if column not in excluded]
    return DataSchema(
        id_column=data_config.id_column,
        date_column=data_config.date_column,
        target_column=None,
        required_columns=required,
        numeric_columns=[
            column for column in data_config.numeric_columns if column not in excluded
        ],
        categorical_columns=[
            column for column in data_config.categorical_columns if column not in excluded
        ],
        bounds={
            column: bound
            for column, bound in data_config.bounds.items()
            if column not in excluded
        },
        allowed_values={
            column: values
            for column, values in data_config.allowed_values.items()
            if column not in excluded
        },
    )
