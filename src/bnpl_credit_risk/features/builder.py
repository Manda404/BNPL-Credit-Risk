"""Configuration-driven feature creation for BNPL credit-risk modelling.

Features available at checkout model affordability, repayment burden,
customer profile and transaction seasonality. Behavioral features are kept
for post-origination monitoring but excluded from the application-risk scope.

The implementation is hardened against common edge cases such as zero income,
invalid installment counts, ages outside configured bins, unparseable dates
and missing behavioral risk scores.

Each `_add_*` method is independently testable and pure (no fitted state),
which is what makes them safe to call identically at train time and at
inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from bnpl_credit_risk.constants import RISK_SCOPE_APPLICATION
from bnpl_credit_risk.exceptions import FeatureEngineeringError
from bnpl_credit_risk.settings import FeaturesConfig

ENGINEERED_FEATURE_COLUMNS = (
    "payment_stress",
    "installment_amount",
    "installment_burden_ratio",
    "affordability_band",
    "income_after_installment",
    "credit_score_band",
    "age_group",
    "installment_term",
    "transaction_month_sin",
    "transaction_month_cos",
    "is_high_risk",
)


class BNPLFeatureBuilder:
    """Deterministic feature engineering, config-driven and leakage-scope-aware."""

    def __init__(
        self,
        features_config: FeaturesConfig,
        risk_scope: str,
        date_column: str = "transaction_date",
    ) -> None:
        self._config = features_config
        self._risk_scope = risk_scope
        self._date_column = date_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        specs = self._config.engineered_features

        if self._is_enabled("payment_stress"):
            df = self._add_payment_stress(df, specs["payment_stress"].requires)
        if self._is_enabled("installment_amount"):
            df = self._add_installment_amount(df, specs["installment_amount"].requires)
        if self._is_enabled("installment_burden_ratio"):
            df = self._add_installment_burden_ratio(
                df, specs["installment_burden_ratio"].requires
            )
        if self._is_enabled("affordability_band"):
            df = self._add_affordability_band(
                df,
                specs["affordability_band"].requires,
                specs["affordability_band"].bins,
                specs["affordability_band"].labels,
            )
        if self._is_enabled("income_after_installment"):
            df = self._add_income_after_installment(
                df, specs["income_after_installment"].requires
            )
        if self._is_enabled("credit_score_band"):
            df = self._add_credit_score_band(
                df,
                specs["credit_score_band"].requires,
                specs["credit_score_band"].bins,
                specs["credit_score_band"].labels,
            )
        if self._is_enabled("age_group"):
            df = self._add_age_group(
                df, specs["age_group"].requires, specs["age_group"].bins, specs["age_group"].labels
            )
        if self._is_enabled("installment_term"):
            df = self._add_installment_term(
                df,
                specs["installment_term"].requires,
                specs["installment_term"].mapping,
            )
        if self._is_enabled("transaction_month_sin") or self._is_enabled(
            "transaction_month_cos"
        ):
            df = self._add_transaction_month_cycle(df)
        if self._is_enabled("is_high_risk"):
            df = self._add_is_high_risk(
                df, specs["is_high_risk"].requires, specs["is_high_risk"].risk_score_threshold
            )

        return df

    def _is_enabled(self, feature_name: str) -> bool:
        spec = self._config.engineered_features[feature_name]
        return spec.enabled and not (
            self._risk_scope == RISK_SCOPE_APPLICATION and spec.leaky
        )

    def _require_columns(self, df: pd.DataFrame, columns: list[str], feature_name: str) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise FeatureEngineeringError(
                f"Cannot compute '{feature_name}': missing source column(s) {missing}"
            )

    def _add_payment_stress(self, df: pd.DataFrame, requires: list[str]) -> pd.DataFrame:
        self._require_columns(df, requires, "payment_stress")
        delay = pd.to_numeric(df["repayment_delay_days"], errors="coerce").clip(lower=0)
        missed = pd.to_numeric(df["missed_payments"], errors="coerce").clip(lower=0)
        df["payment_stress"] = (delay * missed).fillna(0.0)
        return df

    def _add_installment_amount(self, df: pd.DataFrame, requires: list[str]) -> pd.DataFrame:
        """Estimate the amount due for each BNPL installment."""
        self._require_columns(df, requires, "installment_amount")
        purchase = pd.to_numeric(df["purchase_amount"], errors="coerce").clip(lower=0)
        installments = pd.to_numeric(df["bnpl_installments"], errors="coerce")
        valid_installments = installments.where(installments > 0)
        df["installment_amount"] = purchase / valid_installments
        return df

    def _add_installment_burden_ratio(
        self, df: pd.DataFrame, requires: list[str]
    ) -> pd.DataFrame:
        """Measure the share of monthly income consumed by one installment."""
        self._require_columns(df, requires, "installment_burden_ratio")
        purchase = pd.to_numeric(df["purchase_amount"], errors="coerce").clip(lower=0)
        installments = pd.to_numeric(df["bnpl_installments"], errors="coerce")
        income = pd.to_numeric(df["monthly_income"], errors="coerce").clip(lower=0)
        installment_amount = purchase / installments.where(installments > 0)
        df["installment_burden_ratio"] = installment_amount / income.where(income > 0)
        return df

    def _add_affordability_band(
        self,
        df: pd.DataFrame,
        requires: list[str],
        bins: list[float] | None,
        labels: list[str] | None,
    ) -> pd.DataFrame:
        """Translate installment burden into an interpretable policy band."""
        self._require_columns(df, requires, "affordability_band")
        if not bins or not labels:
            raise FeatureEngineeringError(
                "affordability_band requires 'bins' and 'labels' in configs/features.yaml"
            )

        burden = pd.to_numeric(df["installment_burden_ratio"], errors="coerce")
        band = pd.cut(burden, bins=bins, labels=labels, include_lowest=True)
        df["affordability_band"] = band.astype(object).fillna("Unknown")
        return df

    def _add_income_after_installment(
        self, df: pd.DataFrame, requires: list[str]
    ) -> pd.DataFrame:
        """Estimate monthly income remaining after one BNPL installment."""
        self._require_columns(df, requires, "income_after_installment")
        income = pd.to_numeric(df["monthly_income"], errors="coerce").clip(lower=0)
        installment = pd.to_numeric(df["installment_amount"], errors="coerce")
        df["income_after_installment"] = income - installment
        return df

    def _add_credit_score_band(
        self,
        df: pd.DataFrame,
        requires: list[str],
        bins: list[float] | None,
        labels: list[str] | None,
    ) -> pd.DataFrame:
        """Convert the bureau score into configurable, interpretable bands."""
        self._require_columns(df, requires, "credit_score_band")
        if not bins or not labels:
            raise FeatureEngineeringError(
                "credit_score_band requires 'bins' and 'labels' in configs/features.yaml"
            )

        credit_score = pd.to_numeric(df["credit_score"], errors="coerce")
        score_band = pd.cut(
            credit_score,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        df["credit_score_band"] = score_band.astype(object).fillna("Unknown")
        return df

    def _add_age_group(
        self, df: pd.DataFrame, requires: list[str], bins: list[float] | None, labels: list[str] | None
    ) -> pd.DataFrame:
        self._require_columns(df, requires, "age_group")
        if not bins or not labels:
            raise FeatureEngineeringError("age_group requires 'bins' and 'labels' in configs/features.yaml")

        age = pd.to_numeric(df["age"], errors="coerce")
        out_of_range = age.notna() & ((age < bins[0]) | (age > bins[-1]))
        if out_of_range.any():
            logger.bind(pipeline="features.builder").warning(
                "{} row(s) have age outside configured bins {} — mapped to 'Unknown' age_group",
                int(out_of_range.sum()),
                bins,
            )

        age_group = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
        df["age_group"] = age_group.astype(object).where(~out_of_range, "Unknown").fillna("Unknown")
        return df

    def _add_installment_term(
        self,
        df: pd.DataFrame,
        requires: list[str],
        mapping: dict[int, str] | None,
    ) -> pd.DataFrame:
        """Map supported installment counts to business-friendly term labels."""
        self._require_columns(df, requires, "installment_term")
        if not mapping:
            raise FeatureEngineeringError(
                "installment_term requires 'mapping' in configs/features.yaml"
            )

        installments = pd.to_numeric(df["bnpl_installments"], errors="coerce")
        df["installment_term"] = installments.map(mapping).fillna("Unknown")
        return df

    def _add_transaction_month_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode month seasonality without creating an artificial 1→12 order."""
        self._require_columns(df, [self._date_column], "transaction_month_cycle")
        parsed = pd.to_datetime(df[self._date_column], errors="coerce")
        month = parsed.dt.month.astype(float)
        invalid = month.isna()
        if invalid.any():
            logger.bind(pipeline="features.builder").warning(
                "{} row(s) have an unparseable {} — cyclical month features set to 0",
                int(invalid.sum()),
                self._date_column,
            )

        angle = 2 * np.pi * month / 12
        if self._is_enabled("transaction_month_sin"):
            df["transaction_month_sin"] = np.sin(angle).where(~invalid, 0.0)
        if self._is_enabled("transaction_month_cos"):
            df["transaction_month_cos"] = np.cos(angle).where(~invalid, 0.0)
        return df

    def _add_is_high_risk(
        self, df: pd.DataFrame, requires: list[str], threshold: float | None
    ) -> pd.DataFrame:
        self._require_columns(df, requires, "is_high_risk")
        if threshold is None:
            raise FeatureEngineeringError("is_high_risk requires 'risk_score_threshold' in configs/features.yaml")
        risk_score = pd.to_numeric(df["risk_score"], errors="coerce")
        # NaN risk_score conservatively resolves to not-high-risk (0) rather than raising —
        # this feature is only used in the behavioral_risk scope where risk_score is required
        # upstream by the data contract, so NaN here means an already-flagged validation issue.
        df["is_high_risk"] = (risk_score > threshold).fillna(False).astype(int)
        return df


def resolve_feature_columns(features_config: FeaturesConfig, risk_scope: str) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for the given risk scope.

    application_risk excludes leaky raw columns and any engineered feature
    marked `leaky: true` in configs/features.yaml. behavioral_risk includes
    everything, reproducing the original notebook's feature set.
    """
    numeric = list(features_config.numeric_features)
    categorical = list(features_config.categorical_features)

    if risk_scope == "behavioral_risk":
        numeric += list(features_config.behavioral_numeric_features)
        categorical += list(features_config.behavioral_categorical_features)
    elif risk_scope == "application_risk":
        leaky_engineered = {
            name for name, spec in features_config.engineered_features.items() if spec.leaky
        }
        numeric = [c for c in numeric if c not in leaky_engineered]
        categorical = [c for c in categorical if c not in leaky_engineered]
    else:
        raise FeatureEngineeringError(f"Unknown risk_scope: {risk_scope}")

    return numeric, categorical
