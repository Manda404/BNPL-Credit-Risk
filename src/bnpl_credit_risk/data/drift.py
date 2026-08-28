"""Dataset drift diagnostics for random and chronological partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bnpl_credit_risk.exceptions import DataValidationError


@dataclass(frozen=True)
class TemporalDriftResult:
    reference: pd.DataFrame
    current: pd.DataFrame
    report: pd.DataFrame
    reference_end: pd.Timestamp
    current_start: pd.Timestamp


@dataclass(frozen=True)
class DataDriftReport:
    """Compute PSI, numeric KS distance and unseen-category rates.

    The reference dataset defines numeric bins and known categories. The
    current dataset is never used to fit those definitions.
    """

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_column: str
    date_column: str
    warning_psi: float = 0.10
    critical_psi: float = 0.25
    n_bins: int = 10

    def build(self, reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """Compare ``current`` with ``reference`` using shared columns."""
        if reference.empty or current.empty:
            raise DataValidationError("Drift analysis requires two non-empty datasets")

        rows: list[dict[str, float | str | int]] = []
        for column in self.numeric_columns:
            if column not in reference.columns or column not in current.columns:
                continue
            reference_values = pd.to_numeric(reference[column], errors="coerce")
            current_values = pd.to_numeric(current[column], errors="coerce")
            psi = self._numeric_psi(reference_values, current_values)
            rows.append(
                self._row(
                    column=column,
                    feature_type="numeric",
                    psi=psi,
                    ks_statistic=self._ks_statistic(reference_values, current_values),
                    unseen_category_rate=0.0,
                    reference_missing_rate=float(reference_values.isna().mean()),
                    current_missing_rate=float(current_values.isna().mean()),
                )
            )

        categorical = [*self.categorical_columns]
        if self.target_column in reference.columns and self.target_column in current.columns:
            categorical.append(self.target_column)
        for column in dict.fromkeys(categorical):
            if column not in reference.columns or column not in current.columns:
                continue
            reference_values = reference[column]
            current_values = current[column]
            known = set(reference_values.dropna().astype(str))
            current_non_missing = current_values.dropna().astype(str)
            unseen_rate = (
                float((~current_non_missing.isin(known)).mean())
                if len(current_non_missing)
                else 0.0
            )
            rows.append(
                self._row(
                    column=column,
                    feature_type="target" if column == self.target_column else "categorical",
                    psi=self._categorical_psi(reference_values, current_values),
                    ks_statistic=np.nan,
                    unseen_category_rate=unseen_rate,
                    reference_missing_rate=float(reference_values.isna().mean()),
                    current_missing_rate=float(current_values.isna().mean()),
                )
            )

        return pd.DataFrame(rows).sort_values(
            ["status_rank", "psi"], ascending=[False, False]
        ).drop(columns="status_rank").reset_index(drop=True)

    def build_temporal(
        self,
        df: pd.DataFrame,
        *,
        recent_fraction: float = 0.20,
    ) -> TemporalDriftResult:
        """Compare the oldest rows with the most recent chronological window."""
        if self.date_column not in df.columns:
            raise DataValidationError(
                f"Temporal drift requires date column '{self.date_column}'"
            )
        if not 0 < recent_fraction < 1:
            raise ValueError("recent_fraction must be strictly between 0 and 1")

        ordered = df.copy()
        ordered[self.date_column] = pd.to_datetime(
            ordered[self.date_column], errors="coerce"
        )
        if ordered[self.date_column].isna().any():
            raise DataValidationError(
                f"Temporal drift found unparseable values in '{self.date_column}'"
            )
        ordered = ordered.sort_values(self.date_column).reset_index(drop=True)
        split_index = int(len(ordered) * (1 - recent_fraction))
        if split_index <= 0 or split_index >= len(ordered):
            raise DataValidationError("Temporal drift produced an empty time window")

        reference = ordered.iloc[:split_index].copy()
        current = ordered.iloc[split_index:].copy()
        return TemporalDriftResult(
            reference=reference,
            current=current,
            report=self.build(reference, current),
            reference_end=pd.Timestamp(reference[self.date_column].max()),
            current_start=pd.Timestamp(current[self.date_column].min()),
        )

    def _row(
        self,
        *,
        column: str,
        feature_type: str,
        psi: float,
        ks_statistic: float,
        unseen_category_rate: float,
        reference_missing_rate: float,
        current_missing_rate: float,
    ) -> dict[str, float | str | int]:
        if psi >= self.critical_psi:
            status, rank = "critical", 2
        elif psi >= self.warning_psi:
            status, rank = "warning", 1
        else:
            status, rank = "stable", 0
        return {
            "feature": column,
            "type": feature_type,
            "psi": float(psi),
            "ks_statistic": float(ks_statistic),
            "unseen_category_rate": float(unseen_category_rate),
            "reference_missing_rate": reference_missing_rate,
            "current_missing_rate": current_missing_rate,
            "status": status,
            "status_rank": rank,
        }

    def _numeric_psi(self, reference: pd.Series, current: pd.Series) -> float:
        reference_clean = reference.dropna()
        current_clean = current.dropna()
        if reference_clean.empty or current_clean.empty:
            return 0.0
        edges = np.unique(
            reference_clean.quantile(np.linspace(0, 1, self.n_bins + 1)).to_numpy()
        )
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        reference_share = pd.cut(
            reference_clean, edges, include_lowest=True
        ).value_counts(normalize=True, sort=False)
        current_share = pd.cut(
            current_clean, edges, include_lowest=True
        ).value_counts(normalize=True, sort=False)
        return self._psi(reference_share, current_share)

    def _categorical_psi(self, reference: pd.Series, current: pd.Series) -> float:
        reference_share = (
            reference.fillna("__MISSING__").astype(str).value_counts(normalize=True)
        )
        current_share = (
            current.fillna("__MISSING__").astype(str).value_counts(normalize=True)
        )
        return self._psi(reference_share, current_share)

    @staticmethod
    def _psi(reference_share: pd.Series, current_share: pd.Series) -> float:
        levels = reference_share.index.union(current_share.index)
        epsilon = 1e-6
        reference_aligned = reference_share.reindex(levels, fill_value=epsilon).clip(
            lower=epsilon
        )
        current_aligned = current_share.reindex(levels, fill_value=epsilon).clip(
            lower=epsilon
        )
        return float(
            (
                (current_aligned - reference_aligned)
                * np.log(current_aligned / reference_aligned)
            ).sum()
        )

    @staticmethod
    def _ks_statistic(reference: pd.Series, current: pd.Series) -> float:
        reference_values = np.sort(reference.dropna().to_numpy(dtype=float))
        current_values = np.sort(current.dropna().to_numpy(dtype=float))
        if not len(reference_values) or not len(current_values):
            return 0.0
        support = np.sort(np.unique(np.concatenate([reference_values, current_values])))
        reference_cdf = np.searchsorted(reference_values, support, side="right") / len(
            reference_values
        )
        current_cdf = np.searchsorted(current_values, support, side="right") / len(
            current_values
        )
        return float(np.max(np.abs(reference_cdf - current_cdf)))
