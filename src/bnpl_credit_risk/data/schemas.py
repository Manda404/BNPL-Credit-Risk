"""Data contract: what a training dataframe vs an inference dataframe must look like.

Two distinct schemas exist because they have genuinely different requirements —
merging them into one would let the target column silently become "optional but
tolerated" everywhere, which is exactly how target leakage sneaks into
production. See docs/data_contract.md.
"""

from __future__ import annotations

from pydantic import BaseModel

from bnpl_credit_risk.settings import BoundConfig, DataConfig


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


def inference_input_schema(data_config: DataConfig) -> DataSchema:
    """Schema for inference data: the target column is neither required nor used.

    If present in the input file it is dropped before scoring — batch inference
    input is typically "applications to score", which by definition have no
    known outcome yet.
    """
    required = [c for c in data_config.required_columns if c != data_config.target_column]
    return DataSchema(
        id_column=data_config.id_column,
        date_column=data_config.date_column,
        target_column=None,
        required_columns=required,
        numeric_columns=data_config.numeric_columns,
        categorical_columns=data_config.categorical_columns,
        bounds=data_config.bounds,
        allowed_values=data_config.allowed_values,
    )
