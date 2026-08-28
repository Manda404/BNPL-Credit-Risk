"""Visual summary of the dictionary produced by :class:`DataQualityReport`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bnpl_credit_risk.visualization.eda import COLORS, save_and_close


def plot_data_quality_report(
    report: dict[str, Any],
    save_path: str | Path | None = None,
    *,
    size: tuple[float, float] = (12, 8),
) -> Figure:
    """Render dataset health, missing values, target balance and dtype mix.

    Parameters
    ----------
    report:
        Output of ``DataQualityReport(...).build(df)``.
    save_path:
        Optional destination. When omitted, the returned figure is displayed
        automatically by Jupyter.
    size:
        Figure width and height in inches.
    """
    n_rows = int(report.get("n_rows", 0))
    n_columns = int(report.get("n_columns", 0))
    n_duplicates = int(report.get("n_duplicate_rows", 0))
    n_missing = int(report.get("n_missing_values_total", 0))

    fig, axes = plt.subplots(2, 2, figsize=size)

    # Headline indicators.
    summary_ax = axes[0, 0]
    summary_ax.axis("off")
    indicators = [
        ("Rows", n_rows, COLORS["main"]),
        ("Columns", n_columns, COLORS["accent"]),
        ("Duplicates", n_duplicates, COLORS["default"] if n_duplicates else COLORS["paid"]),
        ("Missing", n_missing, COLORS["default"] if n_missing else COLORS["paid"]),
    ]
    for index, (label, value, color) in enumerate(indicators):
        x = 0.08 + (index % 2) * 0.48
        y = 0.72 - (index // 2) * 0.48
        summary_ax.text(x, y, f"{value:,}", fontsize=24, fontweight="bold", color=color)
        summary_ax.text(x, y - 0.13, label, fontsize=11, color="#4A4A4A")
    summary_ax.set_title("Dataset health", fontweight="bold", loc="left")

    # Missing values by column.
    missing_ax = axes[0, 1]
    missing_by_column = report.get("missing_values_by_column", {})
    if missing_by_column:
        ordered_missing = sorted(missing_by_column.items(), key=lambda item: item[1])
        labels, values = zip(*ordered_missing, strict=False)
        missing_ax.barh(labels, values, color=COLORS["default"])
        missing_ax.set_xlabel("Missing values")
    else:
        missing_ax.axis("off")
        missing_ax.text(
            0.5,
            0.5,
            "No missing values",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=COLORS["paid"],
        )
    missing_ax.set_title("Missing values by column", fontweight="bold")

    # Target distribution, when the report was built with a target column.
    target_ax = axes[1, 0]
    target_distribution = report.get("target_distribution", {})
    if target_distribution:
        target_labels = list(target_distribution)
        target_counts = [int(target_distribution[label]["count"]) for label in target_labels]
        colors = [COLORS["paid"], COLORS["default"]]
        bars = target_ax.bar(target_labels, target_counts, color=colors[: len(target_labels)])
        total = sum(target_counts)
        for bar, count in zip(bars, target_counts, strict=False):
            rate = count / total if total else 0.0
            target_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({rate:.1%})",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        target_ax.set_ylim(0, max(target_counts) * 1.18 if target_counts else 1)
        target_ax.set_xlabel("Target class")
        target_ax.set_ylabel("Rows")
    else:
        target_ax.axis("off")
        target_ax.text(0.5, 0.5, "Target not included", ha="center", va="center")
    target_ax.set_title("Target distribution", fontweight="bold")

    # Compact view of the dataframe's dtype composition.
    dtype_ax = axes[1, 1]
    dtype_counts: dict[str, int] = {}
    for dtype in report.get("dtypes", {}).values():
        dtype_counts[str(dtype)] = dtype_counts.get(str(dtype), 0) + 1
    if dtype_counts:
        dtype_labels = list(dtype_counts)
        dtype_values = [dtype_counts[label] for label in dtype_labels]
        bars = dtype_ax.bar(dtype_labels, dtype_values, color=COLORS["main"])
        for bar, value in zip(bars, dtype_values, strict=False):
            dtype_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        dtype_ax.set_ylim(0, max(dtype_values) * 1.15 if dtype_values else 1)
        dtype_ax.tick_params(axis="x", rotation=20)
        dtype_ax.set_ylabel("Columns")
    else:
        dtype_ax.axis("off")
        dtype_ax.text(0.5, 0.5, "No dtype information", ha="center", va="center")
    dtype_ax.set_title("Column types", fontweight="bold")

    fig.suptitle("Data Quality Report", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
