"""Reusable figures for the three-model boosting benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from bnpl_credit_risk.models.boosting import (
    BoostingBenchmarkResult,
    BoostingModelResult,
)


class BoostingBenchmarkVisualizer:
    """Build consistent figures for every model without notebook plot code."""

    def __init__(self, *, size: tuple[float, float]) -> None:
        self._size = size

    def learning_curves(self, result: BoostingModelResult) -> Figure:
        length = min(len(history.train) for history in result.histories)
        train = np.array([history.train[:length] for history in result.histories])
        validation = np.array(
            [history.validation[:length] for history in result.histories]
        )
        rounds = np.arange(1, length + 1)
        fig, ax = plt.subplots(figsize=self._size)
        for curve in train:
            ax.plot(rounds, curve, color="#4C78A8", alpha=0.15)
        for curve in validation:
            ax.plot(rounds, curve, color="#E45756", alpha=0.15)
        ax.plot(
            rounds,
            train.mean(axis=0),
            color="#4C78A8",
            linewidth=2.5,
            label="Train mean",
        )
        ax.plot(
            rounds,
            validation.mean(axis=0),
            color="#E45756",
            linewidth=2.5,
            label="Validation mean",
        )
        ax.set(
            title=f"Cross-validated Learning Curves — {result.model_name}",
            xlabel="Boosting iteration",
            ylabel=result.histories[0].metric_name,
        )
        ax.legend()
        ax.grid(alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def oof_performance(
        self,
        y_true: pd.Series | np.ndarray,
        result: BoostingModelResult,
        *,
        threshold: float = 0.5,
    ) -> Figure:
        probability = result.oof_probability
        prediction = (probability >= threshold).astype(int)
        fig, axes = plt.subplots(1, 2, figsize=self._size)
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        ConfusionMatrixDisplay(
            matrix, display_labels=["Paid", "Defaulted"]
        ).plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title(
            f"OOF Confusion Matrix — {result.model_name} — threshold={threshold:.2f}",
            fontweight="bold",
        )

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.5,
            label=f"{result.model_name} OOF (AUC={auc:.3f})",
        )
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[1].set(
            title=f"OOF ROC Curve — {result.model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def metric_summary(self, result: BoostingModelResult) -> Figure:
        summary = result.fold_metrics.agg(["mean", "std"]).T
        positions = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=self._size)
        ax.bar(positions, summary["mean"], yerr=summary["std"], capsize=4)
        ax.set_xticks(positions, summary.index, rotation=25, ha="right")
        ax.set(
            title=f"Cross-validation Metrics — {result.model_name}",
            ylabel="Fold mean ± standard deviation",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def comparison(self, benchmark: BoostingBenchmarkResult) -> Figure:
        metrics = ["ROC-AUC mean", "PR-AUC mean", "MCC mean"]
        frame = benchmark.comparison.loc[:, metrics]
        ax = frame.plot(kind="bar", figsize=self._size, width=0.75)
        ax.set(
            title="Out-of-fold Model Comparison",
            xlabel="Model",
            ylabel="Mean cross-validation score",
            ylim=(0, 1),
        )
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.20)
        ax.legend([metric.removesuffix(" mean") for metric in metrics])
        fig = ax.figure
        fig.tight_layout()
        plt.close(fig)
        return fig
