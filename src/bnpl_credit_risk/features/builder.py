"""BNPLFeatureBuilder — reproduces the five engineered features from the
notebook's Step 4 (payment_stress, income_to_purchase_ratio, age_group,
txn_month, is_high_risk), hardened against the edge cases the notebook never
had to face because the raw dataset happens to be clean:

- division by zero / negative amounts in income_to_purchase_ratio
- ages outside the fixed bin range
- unparseable / missing transaction_date
- missing risk_score

Each `_add_*` method is independently testable and pure (no fitted state),
which is what makes them safe to call identically at train time and at
inference time.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from bnpl_credit_risk.exceptions import FeatureEngineeringError
from bnpl_credit_risk.settings import FeaturesConfig

ENGINEERED_FEATURE_COLUMNS = (
    "payment_stress",
    "income_to_purchase_ratio",
    "age_group",
    "txn_month",
    "is_high_risk",
)


class BNPLFeatureBuilder:
    """Deterministic feature engineering, config-driven and leakage-scope-aware."""

    def __init__(self, features_config: FeaturesConfig, date_column: str = "transaction_date") -> None:
        self._config = features_config
        self._date_column = date_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        specs = self._config.engineered_features

        if specs["payment_stress"].enabled:
            df = self._add_payment_stress(df, specs["payment_stress"].requires)
        if specs["income_to_purchase_ratio"].enabled:
            df = self._add_income_to_purchase_ratio(df, specs["income_to_purchase_ratio"].requires)
        if specs["age_group"].enabled:
            df = self._add_age_group(
                df, specs["age_group"].requires, specs["age_group"].bins, specs["age_group"].labels
            )
        if specs["txn_month"].enabled:
            df = self._add_txn_month(df, specs["txn_month"].requires)
        if specs["is_high_risk"].enabled:
            df = self._add_is_high_risk(
                df, specs["is_high_risk"].requires, specs["is_high_risk"].risk_score_threshold
            )

        return df

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

    def _add_income_to_purchase_ratio(self, df: pd.DataFrame, requires: list[str]) -> pd.DataFrame:
        self._require_columns(df, requires, "income_to_purchase_ratio")
        income = pd.to_numeric(df["monthly_income"], errors="coerce").clip(lower=0)
        purchase = pd.to_numeric(df["purchase_amount"], errors="coerce").clip(lower=0)
        # +1 denominator floor avoids division by zero even if purchase_amount is 0.
        df["income_to_purchase_ratio"] = income / (purchase + 1)
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

    def _add_txn_month(self, df: pd.DataFrame, requires: list[str]) -> pd.DataFrame:
        self._require_columns(df, requires, "txn_month")
        parsed = pd.to_datetime(df[self._date_column], errors="coerce")
        invalid = parsed.isna()
        if invalid.any():
            logger.bind(pipeline="features.builder").warning(
                "{} row(s) have an unparseable {} — txn_month set to 0",
                int(invalid.sum()),
                self._date_column,
            )
        df["txn_month"] = parsed.dt.month.fillna(0).astype(int)
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
