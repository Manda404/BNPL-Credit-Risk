"""Figures for threshold, calibration, lift and native TreeSHAP diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from bnpl_credit_risk.models.boosting import ShapExplanation


class ModelDiagnosticsVisualizer:
    def __init__(self, *, size: tuple[float, float]) -> None:
        self._size = size

    def calibration(
        self,
        table: pd.DataFrame,
        probability: np.ndarray,
        model_name: str,
    ) -> Figure:
        populated = table.loc[table["Applications"] > 0]
        fig, axes = plt.subplots(1, 2, figsize=self._size)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        axes[0].plot(
            populated["Mean_probability"],
            populated["Observed_default_rate"],
            marker="o",
            color="#378ADD",
            label=model_name,
        )
        axes[0].set(
            title=f"OOF Calibration — {model_name}",
            xlabel="Mean predicted default probability",
            ylabel="Observed default rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[0].legend()
        axes[0].grid(alpha=0.20)

        axes[1].hist(
            probability,
            bins=20,
            color="#378ADD",
            edgecolor="white",
            alpha=0.85,
        )
        axes[1].set(
            title="OOF Probability Distribution",
            xlabel="P(default)",
            ylabel="Applications",
        )
        axes[1].grid(axis="y", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def classification_performance(
        self,
        y_true: np.ndarray,
        probability: np.ndarray,
        model_name: str,
        *,
        threshold: float,
        dataset_label: str,
    ) -> Figure:
        prediction = (probability >= threshold).astype(int)
        fig, axes = plt.subplots(1, 2, figsize=self._size)
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        ConfusionMatrixDisplay(
            matrix, display_labels=["Paid", "Defaulted"]
        ).plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title(
            f"{dataset_label} Confusion Matrix — threshold={threshold:.3f}",
            fontweight="bold",
        )

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.5,
            label=f"{model_name} (AUC={auc:.3f})",
        )
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[1].set(
            title=f"{dataset_label} ROC Curve — {model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def threshold_tradeoff(
        self,
        grid: pd.DataFrame,
        selected_threshold: float,
        model_name: str,
    ) -> Figure:
        ordered = grid.sort_values("Threshold")
        fig, axes = plt.subplots(1, 2, figsize=self._size)
        axes[0].plot(
            ordered["Threshold"],
            ordered["Recall"],
            marker="o",
            color="#C44E52",
            label="Recall — defaults caught",
        )
        axes[0].plot(
            ordered["Threshold"],
            ordered["Precision"],
            marker="s",
            color="#4C72B0",
            label="Precision — alerts confirmed",
        )
        axes[0].set(
            title=f"Precision/Recall Trade-off — {model_name} (OOF)",
            xlabel="Decision threshold",
            ylabel="Score",
            ylim=(0, 1.02),
        )

        axes[1].plot(
            ordered["Threshold"],
            ordered["Defaults caught"],
            marker="o",
            color="#C44E52",
            label="Defaults caught (TP)",
        )
        axes[1].plot(
            ordered["Threshold"],
            ordered["Defaults missed"],
            marker="^",
            color="#DD8452",
            label="Defaults missed (FN)",
        )
        axes[1].plot(
            ordered["Threshold"],
            ordered["False positive alerts"],
            marker="s",
            color="#4C72B0",
            label="False alerts (FP)",
        )
        axes[1].set(
            title=f"Operational Impact — {model_name} (OOF)",
            xlabel="Decision threshold",
            ylabel="Applications",
        )

        for axis in axes:
            axis.axvline(
                selected_threshold,
                color="black",
                linestyle="--",
                label=f"Selected threshold = {selected_threshold:.3f}",
            )
            axis.grid(alpha=0.20)
            axis.legend()
        fig.tight_layout()
        plt.close(fig)
        return fig

    def lift_and_gains(self, table: pd.DataFrame, model_name: str) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=self._size)
        axes[0].plot(
            table["Cumulative_population"],
            table["Cumulative_defaults_caught"],
            marker="o",
            linewidth=2.5,
            label=model_name,
        )
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[0].set(
            title="Cumulative Gains Curve",
            xlabel="Cumulative share of applications reviewed",
            ylabel="Cumulative share of defaults caught",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axes[0].legend()

        axes[1].bar(
            table["Risk group"].astype(str),
            table["Lift"],
            color="#378ADD",
        )
        axes[1].axhline(1, color="black", linestyle="--", alpha=0.5)
        axes[1].set(
            title="Lift by Risk Decile (1 = highest risk)",
            xlabel="Risk group",
            ylabel="Lift versus portfolio default rate",
        )
        for axis in axes:
            axis.grid(alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def feature_importance(
        self,
        importances: pd.Series,
        model_name: str,
        *,
        top_n: int = 15,
    ) -> Figure:
        top = importances.sort_values().tail(top_n)
        fig, ax = plt.subplots(figsize=self._size)
        bars = ax.barh(top.index, top.values, color="#378ADD")
        for bar, value in zip(bars, top.values, strict=False):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}",
                va="center",
            )
        ax.set(
            title=f"Native Feature Importance — {model_name}",
            xlabel="Model importance",
            ylabel="",
        )
        ax.grid(axis="x", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def shap_global_bar(
        self, explanation: ShapExplanation, model_name: str, *, top_n: int = 12
    ) -> Figure:
        importance = explanation.values.abs().mean().sort_values().tail(top_n)
        fig, ax = plt.subplots(figsize=self._size)
        ax.barh(importance.index, importance.values, color="#BA7517")
        ax.set(
            title=f"Global SHAP Importance — {model_name}",
            xlabel="Mean absolute SHAP contribution",
            ylabel="",
        )
        ax.grid(axis="x", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def shap_beeswarm(
        self, explanation: ShapExplanation, model_name: str, *, top_n: int = 12
    ) -> Figure:
        top_features = (
            explanation.values.abs().mean().sort_values(ascending=False).head(top_n).index
        )
        fig, ax = plt.subplots(figsize=self._size)
        random = np.random.default_rng(42)
        for position, feature in enumerate(reversed(top_features)):
            raw = explanation.features[feature]
            if pd.api.types.is_numeric_dtype(raw):
                color_value = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
            else:
                color_value = pd.Categorical(raw.astype("string")).codes.astype(float)
            jitter = random.normal(0, 0.09, len(raw))
            scatter = ax.scatter(
                explanation.values[feature],
                position + jitter,
                c=color_value,
                cmap="coolwarm",
                alpha=0.65,
                s=18,
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(top_features)), list(reversed(top_features)))
        ax.set(
            title=f"SHAP Summary — {model_name}",
            xlabel="SHAP contribution to default risk",
            ylabel="",
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        colorbar.set_label("Feature value (low → high / category code)")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def shap_waterfall(
        self,
        explanation: ShapExplanation,
        model_name: str,
        *,
        row_position: int = 0,
        top_n: int = 12,
    ) -> Figure:
        values = explanation.values.iloc[row_position]
        features = explanation.features.iloc[row_position]
        selected = values.abs().nlargest(top_n).index
        ordered = values.loc[selected].sort_values()
        labels = [f"{name} = {features[name]}" for name in ordered.index]
        colors = np.where(ordered.values >= 0, "#E24B4A", "#378ADD")
        fig, ax = plt.subplots(figsize=self._size)
        ax.barh(labels, ordered.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=f"Local SHAP Explanation — {model_name} — row {features.name}",
            xlabel="Contribution to model raw default score",
            ylabel="",
        )
        ax.grid(axis="x", alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def shap_dependence(
        self,
        explanation: ShapExplanation,
        feature: str,
        model_name: str,
    ) -> Figure:
        raw = explanation.features[feature]
        fig, ax = plt.subplots(figsize=self._size)
        if pd.api.types.is_numeric_dtype(raw):
            x = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        else:
            categorical = pd.Categorical(raw.astype("string"))
            x = categorical.codes
            ax.set_xticks(range(len(categorical.categories)), categorical.categories, rotation=30)
        ax.scatter(x, explanation.values[feature], alpha=0.55, color="#378ADD")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            title=f"SHAP Dependence — {feature} — {model_name}",
            xlabel=feature,
            ylabel="SHAP contribution to default risk",
        )
        ax.grid(alpha=0.20)
        fig.tight_layout()
        plt.close(fig)
        return fig
