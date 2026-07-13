"""EDA plotting functions — reproduce the notebook's Step 3 visualizations as
reusable functions instead of inline cell code.

Every function returns a `matplotlib.figure.Figure` and never calls
`plt.show()`, so the same function works in a notebook (where the figure
auto-displays) and in an automated pipeline (where the caller saves and
closes it via `save_and_close`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

COLORS = {"default": "#E24B4A", "paid": "#1D9E75", "main": "#378ADD", "accent": "#BA7517"}


def save_and_close(fig: Figure, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_target_distribution(
    df: pd.DataFrame, target_column: str = "default_flag", save_path: str | Path | None = None
) -> Figure:
    counts = df[target_column].value_counts().sort_index()
    labels = ["Paid on Time", "Defaulted"]
    colors = [COLORS["paid"], COLORS["default"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Default vs Paid — Overall Split", fontweight="bold")
    bars = axes[1].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values, strict=False):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", fontweight="bold")
    axes[1].set_title("Count of Each Class", fontweight="bold")
    fig.suptitle(f"Target Variable: {target_column}", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_numeric_by_target(
    df: pd.DataFrame,
    numeric_column: str,
    target_column: str = "default_flag",
    kind: str = "hist",
    save_path: str | Path | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    if kind == "hist":
        for flag, label, color in [(0, "Paid", COLORS["paid"]), (1, "Defaulted", COLORS["default"])]:
            subset = df.loc[df[target_column] == flag, numeric_column]
            ax.hist(subset, bins=20, alpha=0.65, label=label, color=color, edgecolor="white")
        ax.legend()
    elif kind == "box":
        df.boxplot(column=numeric_column, by=target_column, ax=ax)
        plt.suptitle("")
    else:
        raise ValueError(f"Unknown kind: {kind}")
    ax.set_title(f"{numeric_column} by {target_column}", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_default_rate_by_category(
    df: pd.DataFrame,
    category_column: str,
    target_column: str = "default_flag",
    save_path: str | Path | None = None,
) -> Figure:
    rate = (df.groupby(category_column)[target_column].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    rate_values = np.asarray(rate.values, dtype=float)
    bars = ax.bar(rate.index.astype(str), rate_values, color=COLORS["main"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, rate_values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}%", ha="center", fontweight="bold")
    ax.set_title(f"Default Rate by {category_column}", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_default_rate_by_installments(
    df: pd.DataFrame,
    installments_column: str = "bnpl_installments",
    target_column: str = "default_flag",
    save_path: str | Path | None = None,
) -> Figure:
    rate = (df.groupby(installments_column)[target_column].mean() * 100).sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rate.index, np.asarray(rate.values, dtype=float), marker="o", color=COLORS["default"], linewidth=2.5, markersize=10)
    ax.set_title("Default Rate by Number of Installments", fontweight="bold")
    ax.set_xlabel(installments_column)
    ax.set_ylabel("Default Rate (%)")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_risk_score_by_segment(
    df: pd.DataFrame,
    segment_column: str = "customer_segment",
    risk_score_column: str = "risk_score",
    segment_order: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    segment_order = segment_order or ["Low Risk", "Medium Risk", "High Risk"]
    data = [
        np.asarray(df.loc[df[segment_column] == seg, risk_score_column].values, dtype=float)
        for seg in segment_order
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=segment_order, patch_artist=True)
    for patch, color in zip(bp["boxes"], [COLORS["paid"], COLORS["accent"], COLORS["default"]], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"{risk_score_column} by {segment_column}", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, numeric_columns: list[str], save_path: str | Path | None = None
) -> Figure:
    """Pure-matplotlib heatmap (no seaborn dependency) of the lower triangle
    of the correlation matrix, annotated with each coefficient."""
    corr = df[numeric_columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    masked = np.ma.masked_where(mask, corr.to_numpy())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticklabels(numeric_columns)
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap — Numeric Features", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_default_rate_by_missed_payments(
    df: pd.DataFrame,
    missed_payments_column: str = "missed_payments",
    target_column: str = "default_flag",
    save_path: str | Path | None = None,
) -> Figure:
    rate = df.groupby(missed_payments_column)[target_column].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(rate.index, np.asarray(rate.values, dtype=float), color=COLORS["accent"], edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Default Rate by Missed Payments", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_default_rate_by_delay_bins(
    df: pd.DataFrame,
    delay_column: str = "repayment_delay_days",
    target_column: str = "default_flag",
    bins: list[float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    bins = bins or [0, 5, 10, 15, 20, 25, 33]
    delay_bins = pd.cut(df[delay_column], bins=bins, include_lowest=True)
    rate = df.groupby(delay_bins, observed=True)[target_column].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(rate)), np.asarray(rate.values, dtype=float), color=COLORS["accent"], edgecolor="white")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([str(i) for i in rate.index], rotation=20, fontsize=9)
    ax.set_title(f"Default Rate by {delay_column}", fontweight="bold")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
