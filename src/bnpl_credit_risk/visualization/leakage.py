"""Object-oriented facade for leakage-focused BNPL exploratory analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from bnpl_credit_risk.visualization.eda import (
    plot_correlation_heatmap,
    plot_target_distribution,
)


class BNPLLeakageAnalysis:
    """Centralize the shared configuration of leakage-analysis figures.

    The low-level plotting functions remain reusable, while notebooks only
    configure the target, numeric columns and figure size once.
    """

    def __init__(
        self,
        *,
        target_column: str,
        numeric_columns: list[str],
        size: tuple[float, float] = (15, 6),
    ) -> None:
        if not target_column:
            raise ValueError("target_column must not be empty")
        if not numeric_columns:
            raise ValueError("numeric_columns must not be empty")
        if any(dimension <= 0 for dimension in size):
            raise ValueError("Figure size dimensions must be strictly positive")

        self._target_column = target_column
        self._numeric_columns = list(
            dict.fromkeys([*numeric_columns, target_column])
        )
        self._size = size

    def show_target_distribution(self, df: pd.DataFrame) -> None:
        """Display the class distribution of the configured target."""
        plot_target_distribution(
            df,
            target_column=self._target_column,
            size=self._size,
        )
        plt.show()

    def show_correlation(self, df: pd.DataFrame) -> None:
        """Display correlations for configured numeric columns and target."""
        plot_correlation_heatmap(
            df,
            numeric_columns=self._numeric_columns,
            size=self._size,
        )
        plt.show()
