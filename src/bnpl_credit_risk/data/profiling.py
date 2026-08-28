"""Reusable column-level profiling for pandas datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

PROFILE_COLUMNS = (
    "Column",
    "Type",
    "Non-Null",
    "Missing",
    "% Missing",
    "Cardinality",
    "% Unique",
    "Zeros",
    "% Zeros",
    "Infinite",
    "Outliers",
    "% Outliers",
    "Min",
    "Mean",
    "Median",
    "Max",
    "Std Dev",
    "Examples",
)


@dataclass(frozen=True)
class DatasetProfileOptions:
    """Configuration for a column-level dataset profile."""

    numeric_examples: int = 5
    categorical_examples: int = 10
    iqr_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.numeric_examples < 1 or self.categorical_examples < 1:
            raise ValueError("Example limits must be positive integers")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be greater than zero")


class DatasetInspector:
    """Build a detailed quality profile without mutating the source dataframe."""

    def __init__(self, options: DatasetProfileOptions | None = None) -> None:
        self._options = options or DatasetProfileOptions()

    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.bind(pipeline="data.profiling").info(
            "Computing dataset quality summary rows={} columns={}",
            len(df),
            len(df.columns),
        )
        rows = [
            self._profile_column(column, df[column], total_rows=len(df))
            for column in df.columns
        ]
        profile = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
        if profile.empty:
            return profile

        return profile.sort_values(
            "% Missing", ascending=False, kind="stable", ignore_index=True
        )

    def _profile_column(
        self, column: str, series: pd.Series, *, total_rows: int
    ) -> dict[str, Any]:
        non_null = int(series.notna().sum())
        missing = total_rows - non_null
        cardinality = int(series.nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        )
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        example_limit = (
            self._options.numeric_examples
            if is_numeric or is_datetime
            else self._options.categorical_examples
        )

        row = self._empty_profile_values()
        row.update(
            {
                "Column": column,
                "Type": str(series.dtype),
                "Non-Null": non_null,
                "Missing": missing,
                "% Missing": self._percentage(missing, total_rows),
                "Cardinality": cardinality,
                "% Unique": self._percentage(cardinality, non_null),
                "Examples": self._examples(series, example_limit),
            }
        )

        if is_numeric:
            row.update(self._numeric_statistics(series))
        elif is_datetime and non_null:
            row.update(self._datetime_statistics(series))

        return row

    @staticmethod
    def _empty_profile_values() -> dict[str, Any]:
        return dict.fromkeys(PROFILE_COLUMNS, pd.NA)

    def _numeric_statistics(self, series: pd.Series) -> dict[str, Any]:
        numeric = pd.to_numeric(series, errors="coerce")
        finite_mask = numeric.notna() & np.isfinite(numeric)
        finite = numeric.loc[finite_mask].astype(float)
        infinite = int((numeric.notna() & ~np.isfinite(numeric)).sum())
        zeros = int((finite == 0).sum())
        statistics: dict[str, Any] = {
            "Zeros": zeros,
            "% Zeros": self._percentage(zeros, len(finite)),
            "Infinite": infinite,
            "Outliers": 0,
            "% Outliers": 0.0,
        }
        if finite.empty:
            return statistics

        q1 = float(finite.quantile(0.25))
        q3 = float(finite.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - self._options.iqr_multiplier * iqr
        upper_fence = q3 + self._options.iqr_multiplier * iqr
        outliers = int(((finite < lower_fence) | (finite > upper_fence)).sum())
        statistics.update(
            {
                "Outliers": outliers,
                "% Outliers": self._percentage(outliers, len(finite)),
                "Min": self._round(finite.min()),
                "Mean": self._round(finite.mean()),
                "Median": self._round(finite.median()),
                "Max": self._round(finite.max()),
                "Std Dev": self._round(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            }
        )
        return statistics

    @staticmethod
    def _datetime_statistics(series: pd.Series) -> dict[str, Any]:
        non_null = series.dropna()
        return {"Min": non_null.min(), "Max": non_null.max()}

    @staticmethod
    def _percentage(numerator: int, denominator: int) -> float:
        return round((numerator / denominator) * 100, 2) if denominator else 0.0

    @staticmethod
    def _round(value: Any) -> float:
        return round(float(value), 4)

    @staticmethod
    def _examples(series: pd.Series, limit: int) -> list[Any]:
        return series.dropna().drop_duplicates().head(limit).tolist()
