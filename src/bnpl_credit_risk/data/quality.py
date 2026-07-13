"""Data quality report — reproduces the notebook's Step 2 health check
(shape, duplicates, nulls, dtypes, target balance) as a reusable, loggable
component instead of ad-hoc `print()` statements.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class DataQualityReport:
    """Computes and serializes a data health summary for a dataframe."""

    def __init__(self, target_column: str | None = None) -> None:
        self._target_column = target_column

    def build(self, df: pd.DataFrame) -> dict[str, Any]:
        report: dict[str, Any] = {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "n_duplicate_rows": int(df.duplicated().sum()),
            "n_missing_values_total": int(df.isnull().sum().sum()),
            "missing_values_by_column": {
                col: int(n) for col, n in df.isnull().sum().items() if n > 0
            },
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        if self._target_column and self._target_column in df.columns:
            counts = df[self._target_column].value_counts()
            total = len(df)
            report["target_distribution"] = {
                str(k): {"count": int(v), "rate": float(v) / total} for k, v in counts.items()
            }

        return report

    def log(self, report: dict[str, Any], *, pipeline: str = "data.quality") -> None:
        log = logger.bind(pipeline=pipeline)
        log.info(
            "Data quality: rows={} columns={} duplicates={} missing_total={}",
            report["n_rows"],
            report["n_columns"],
            report["n_duplicate_rows"],
            report["n_missing_values_total"],
        )
        if report.get("missing_values_by_column"):
            log.warning("Missing values by column: {}", report["missing_values_by_column"])
        if "target_distribution" in report:
            log.info("Target distribution: {}", report["target_distribution"])
