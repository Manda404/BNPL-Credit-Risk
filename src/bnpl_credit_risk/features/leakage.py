"""Semantic leakage audit and removal for BNPL application-risk datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL
from bnpl_credit_risk.exceptions import FeatureEngineeringError
from bnpl_credit_risk.settings import FeaturesConfig


@dataclass(frozen=True)
class LeakageGuardResult:
    """Leakage audit plus a defensive copy of the cleaned dataframe."""

    cleaned: pd.DataFrame
    report: pd.DataFrame
    removed_columns: tuple[str, ...]


class BNPLLeakageGuard:
    """Identify and remove features unavailable at the prediction timestamp."""

    def __init__(self, features_config: FeaturesConfig, risk_scope: str) -> None:
        if risk_scope not in {RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL}:
            raise FeatureEngineeringError(f"Unknown risk_scope: {risk_scope}")
        self._config = features_config
        self._risk_scope = risk_scope

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the configured semantic audit for the supplied dataframe."""
        rows = []
        for column, reason in self._config.leaky_raw_columns.items():
            present = column in df.columns
            should_remove = self._risk_scope == RISK_SCOPE_APPLICATION and present
            rows.append(
                {
                    "Column": column,
                    "Present": present,
                    "Availability": "Post-origination / unproven at checkout",
                    "Leakage reason": reason,
                    "Decision": "Remove" if should_remove else "Retain for behavioral scope",
                }
            )
        return pd.DataFrame(rows)

    def apply(self, df: pd.DataFrame) -> LeakageGuardResult:
        """Audit ``df`` and remove configured columns for application risk."""
        report = self.analyze(df)
        if self._risk_scope == RISK_SCOPE_APPLICATION:
            removed = tuple(
                column
                for column in self._config.leaky_raw_columns
                if column in df.columns
            )
        else:
            removed = ()
        cleaned = df.drop(columns=list(removed), errors="ignore").copy()
        return LeakageGuardResult(
            cleaned=cleaned,
            report=report,
            removed_columns=removed,
        )
