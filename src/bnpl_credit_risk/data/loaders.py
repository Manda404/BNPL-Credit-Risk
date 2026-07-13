"""Raw data loading.

Reproduces the notebook's `pd.read_csv(...)` + `pd.to_datetime(...)` (cell 2),
with the hardcoded Kaggle path replaced by a configurable one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from bnpl_credit_risk.exceptions import ConfigError
from bnpl_credit_risk.settings import DataConfig, Settings


class BNPLDataLoader:
    """Loads the BNPL dataset from CSV and parses the transaction date column."""

    def __init__(self, settings: Settings, data_config: DataConfig) -> None:
        self._settings = settings
        self._data_config = data_config

    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        resolved = self._settings.resolve(str(path)) if path is not None else self._settings.resolve(
            self._settings.data_raw_path
        )
        if not resolved.exists():
            raise ConfigError(f"Raw dataset not found at {resolved}")

        df = pd.read_csv(resolved)
        date_col = self._data_config.date_column
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        logger.bind(pipeline="data.loaders").info(
            "Loaded raw dataset path={} rows={} columns={}",
            resolved,
            df.shape[0],
            df.shape[1],
        )
        return df
