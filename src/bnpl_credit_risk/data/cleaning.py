"""Lightweight, deterministic cleaning applied before validation/feature engineering.

The raw BNPL dataset has zero missing values and zero duplicate rows (verified
during the notebook audit), so this module intentionally stays small — it
normalizes string categories and coerces dtypes defensively for future data
that may not be as clean, rather than implementing imputation strategies that
nothing in the current dataset exercises.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from bnpl_credit_risk.settings import DataConfig


class BNPLDataCleaner:
    """Normalizes dtypes and categorical string formatting; drops exact duplicate rows."""

    def __init__(self, data_config: DataConfig) -> None:
        self._data_config = data_config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        log = logger.bind(pipeline="data.cleaning")
        df = df.copy()

        n_before = len(df)
        df = df.drop_duplicates()
        if len(df) < n_before:
            log.warning("Dropped {} exact duplicate rows", n_before - len(df))

        for col in self._data_config.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        for col in self._data_config.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        date_col = self._data_config.date_column
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        log.info("Cleaned dataframe rows={} columns={}", df.shape[0], df.shape[1])
        return df
