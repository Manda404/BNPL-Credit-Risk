"""Decision, calibration and ranking diagnostics for BNPL default models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from bnpl_credit_risk.models.boosting import classification_metrics
from bnpl_credit_risk.settings import CostMatrixConfig


class ThresholdTradeoffAnalyzer:
    """Translate decision thresholds into operational BNPL consequences."""

    def __init__(self, cost_matrix: CostMatrixConfig) -> None:
        self._cost_matrix = cost_matrix

    def build(
        self,
        y_true: np.ndarray,
        probability: np.ndarray,
        thresholds: np.ndarray,
        selected_threshold: float,
    ) -> pd.DataFrame:
        rows = []
        candidates = np.append(np.asarray(thresholds, dtype=float), selected_threshold)
        for threshold in sorted(np.unique(candidates), reverse=True):
            prediction = (probability >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_true, prediction, labels=[0, 1]
            ).ravel()
            rows.append(
                {
                    "Threshold": threshold,
                    "Defaults caught": tp,
                    "Defaults missed": fn,
                    "False positive alerts": fp,
                    "Precision": tp / (tp + fp) if (tp + fp) else 0.0,
                    "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "False positive rate": fp / (fp + tn) if (fp + tn) else 0.0,
                    "Estimated cost": (
                        fn * self._cost_matrix.false_negative_cost
                        + fp * self._cost_matrix.false_positive_cost
                    ),
                }
            )

        grid = pd.DataFrame(rows)
        grid["Additional defaults caught"] = (
            grid["Defaults caught"].diff().fillna(0).astype(int)
        )
        grid["Additional false alerts"] = (
            grid["False positive alerts"].diff().fillna(0).astype(int)
        )
        grid["Defaults caught per 100 extra false alerts"] = np.where(
            grid["Additional false alerts"] > 0,
            100
            * grid["Additional defaults caught"]
            / grid["Additional false alerts"],
            np.nan,
        )
        grid["Selected"] = np.isclose(grid["Threshold"], selected_threshold)
        return grid


def calibration_table(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Observed default rate versus predicted probability in fixed bins."""
    frame = pd.DataFrame({"Actual": y_true, "Probability": probability})
    frame["Bin"] = pd.cut(
        frame["Probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )
    table = (
        frame.groupby("Bin", observed=False)
        .agg(
            Applications=("Actual", "size"),
            Mean_probability=("Probability", "mean"),
            Observed_default_rate=("Actual", "mean"),
        )
        .reset_index()
    )
    table["Calibration_gap"] = (
        table["Observed_default_rate"] - table["Mean_probability"]
    )
    table["Absolute_gap"] = table["Calibration_gap"].abs()
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    total = table["Applications"].sum()
    if not total:
        return 0.0
    return float(
        (table["Applications"] / total * table["Absolute_gap"].fillna(0)).sum()
    )


def lift_and_gains_table(
    y_true: np.ndarray, probability: np.ndarray, n_groups: int = 10
) -> pd.DataFrame:
    """Build descending risk groups for lift and cumulative-gains analysis."""
    frame = pd.DataFrame({"Actual": y_true, "Probability": probability}).sort_values(
        "Probability", ascending=False
    )
    group = np.ceil(np.arange(1, len(frame) + 1) * n_groups / len(frame)).astype(int)
    frame["Risk group"] = np.clip(group, 1, n_groups)
    table = (
        frame.groupby("Risk group", sort=True)
        .agg(
            Applications=("Actual", "size"),
            Defaults=("Actual", "sum"),
            Mean_probability=("Probability", "mean"),
        )
        .reset_index()
    )
    overall_rate = float(frame["Actual"].mean())
    total_defaults = max(float(frame["Actual"].sum()), 1.0)
    table["Default_rate"] = table["Defaults"] / table["Applications"]
    table["Lift"] = table["Default_rate"] / overall_rate if overall_rate else 0.0
    table["Cumulative_population"] = table["Applications"].cumsum() / len(frame)
    table["Cumulative_defaults_caught"] = (
        table["Defaults"].cumsum() / total_defaults
    )
    table["Cumulative_lift"] = (
        table["Cumulative_defaults_caught"] / table["Cumulative_population"]
    )
    return table


def evaluation_summary(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    metrics = classification_metrics(y_true, probability, threshold)
    metrics["Expected calibration error"] = expected_calibration_error(calibration)
    return pd.DataFrame([metrics], index=["OOF selected model"])
