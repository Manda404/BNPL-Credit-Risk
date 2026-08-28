"""Portfolio-level figures for batch inference outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


class BatchInferenceVisualizer:
    def __init__(self, *, size: tuple[float, float]) -> None:
        self._size = size

    def portfolio_overview(
        self,
        predictions: pd.DataFrame,
        *,
        threshold: float,
        risk_band_order: list[str],
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=self._size)

        band_counts = (
            predictions["risk_band"]
            .value_counts()
            .reindex(risk_band_order, fill_value=0)
        )
        axes[0].bar(band_counts.index, band_counts.values, color="#378ADD")
        axes[0].set(
            title="Risk Band Distribution",
            xlabel="Risk band",
            ylabel="Applications",
        )
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].hist(
            predictions["default_probability"],
            bins=15,
            color="#BA7517",
            edgecolor="white",
        )
        axes[1].axvline(
            threshold,
            color="black",
            linestyle="--",
            label=f"Threshold = {threshold:.3f}",
        )
        axes[1].set(
            title="Predicted Default Probabilities",
            xlabel="P(default)",
            ylabel="Applications",
        )
        axes[1].legend()

        decision_counts = (
            predictions["predicted_default"]
            .map({0: "Not flagged", 1: "Flagged"})
            .value_counts()
            .reindex(["Not flagged", "Flagged"], fill_value=0)
        )
        axes[2].bar(
            decision_counts.index,
            decision_counts.values,
            color=["#1D9E75", "#E24B4A"],
        )
        axes[2].set(
            title="Decision Distribution",
            xlabel="Model decision",
            ylabel="Applications",
        )

        for axis in axes:
            axis.grid(axis="y", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig
