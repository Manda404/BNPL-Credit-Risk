"""Calibration diagnostics — is P(default)=0.3 actually observed ~30% of the time?"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve

from bnpl_credit_risk.visualization.eda import COLORS, save_and_close


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
    save_path: str | Path | None = None,
) -> Figure:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.plot(mean_pred, fraction_pos, marker="o", color=COLORS["main"], label=label)
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_calibration_comparison(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    for y_prob, label, color in [
        (y_prob_uncalibrated, "Before calibration", COLORS["default"]),
        (y_prob_calibrated, "After calibration", COLORS["paid"]),
    ]:
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, fraction_pos, marker="o", color=color, label=label)
    ax.set_title("Calibration — Before vs After", fontweight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
