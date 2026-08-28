"""Visual comparison of target distributions across dataset partitions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from bnpl_credit_risk.exceptions import DataValidationError
from bnpl_credit_risk.visualization.eda import COLORS, save_and_close


def plot_split_target_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_column: str,
    size: tuple[float, float] = (15, 6),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot target-class bars grouped separately under Train and Test."""
    for split_name, frame in (("train", train_df), ("test", test_df)):
        if target_column not in frame.columns:
            raise DataValidationError(
                f"Target column '{target_column}' is missing from {split_name} split"
            )
        if frame.empty:
            raise DataValidationError(f"Cannot plot an empty {split_name} split")

    train_counts = train_df[target_column].value_counts().sort_index()
    test_counts = test_df[target_column].value_counts().sort_index()
    classes = train_counts.index.union(test_counts.index).sort_values()
    train_counts = train_counts.reindex(classes, fill_value=0)
    test_counts = test_counts.reindex(classes, fill_value=0)
    train_rates = train_counts / len(train_df) * 100
    test_rates = test_counts / len(test_df) * 100

    group_positions = np.arange(2)
    group_width = 0.72
    width = group_width / len(classes)
    offsets = (np.arange(len(classes)) - (len(classes) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=size)
    class_colors = [COLORS["paid"], COLORS["default"], COLORS["main"], COLORS["accent"]]
    for class_index, target_class in enumerate(classes):
        counts = [train_counts.loc[target_class], test_counts.loc[target_class]]
        rates = [train_rates.loc[target_class], test_rates.loc[target_class]]
        bars = ax.bar(
            group_positions + offsets[class_index],
            rates,
            width,
            label=f"Class {target_class}",
            color=class_colors[class_index % len(class_colors)],
            edgecolor="white",
        )
        for bar, count, rate in zip(bars, counts, rates, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count):,}\n{float(rate):.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    highest_rate = max(float(train_rates.max()), float(test_rates.max()))
    ax.set_ylim(0, max(100.0, highest_rate * 1.18))
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"Train\n(n={len(train_df):,})", f"Test\n(n={len(test_df):,})"])
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Share of split (%)")
    ax.set_title(
        f"Target Distribution — {target_column} — Stratified Train/Test Split",
        fontweight="bold",
    )
    ax.legend(title="Target class")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
