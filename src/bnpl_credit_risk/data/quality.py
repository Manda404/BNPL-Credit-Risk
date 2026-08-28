"""Data quality report — reproduces the notebook's Step 2 health check
(shape, duplicates, nulls, dtypes, target balance) as a reusable, loggable
component instead of ad-hoc `print()` statements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from bnpl_credit_risk.data.profiling import DatasetInspector, DatasetProfileOptions


class DataQualityReport:
    """Computes and serializes a data health summary for a dataframe."""

    def __init__(
        self,
        target_column: str | None = None,
        size: tuple[float, float] = (12, 8),
    ) -> None:
        if any(dimension <= 0 for dimension in size):
            raise ValueError("DataQualityReport size dimensions must be strictly positive")
        self._target_column = target_column
        self._size = size

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

    def profile_dataset(
        self,
        df: pd.DataFrame,
        *,
        numeric_examples: int = 5,
        categorical_examples: int = 10,
        iqr_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """Return a detailed, column-level quality profile.

        Numeric outliers are detected with Tukey's IQR rule: values below
        ``Q1 - 1.5 * IQR`` or above ``Q3 + 1.5 * IQR``. ``% Outliers`` uses
        the number of finite numeric observations as its denominator.

        Parameters
        ----------
        df:
            Dataset to profile.
        numeric_examples:
            Maximum representative values shown for numeric/date columns.
        categorical_examples:
            Maximum representative values shown for categorical columns.
        iqr_multiplier:
            Tukey multiplier used to define numeric outlier fences.
        """
        options = DatasetProfileOptions(
            numeric_examples=numeric_examples,
            categorical_examples=categorical_examples,
            iqr_multiplier=iqr_multiplier,
        )
        return DatasetInspector(options).inspect(df)

    def show(self, df: pd.DataFrame, save_path: str | Path | None = None) -> None:
        """Build and display the visual quality dashboard for ``df``.

        The plotting dependency is imported lazily so pipelines that only use
        :meth:`build` or :meth:`log` do not need to initialize Matplotlib. The
        method returns ``None`` so Jupyter does not render the figure twice.
        """
        import matplotlib.pyplot as plt

        from bnpl_credit_risk.visualization.quality import plot_data_quality_report

        plot_data_quality_report(
            self.build(df),
            size=self._size,
            save_path=save_path,
        )
        if save_path is None:
            plt.show()

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
