"""BNPLDataValidator — enforces the data contract defined by a DataSchema.

Two categories of problems are treated differently:

- Structural problems (a required column is entirely missing, the id/date
  columns can't be parsed) always raise `DataValidationError`, regardless of
  policy — there is no reasonable way to "quarantine" a missing column.
- Row-level problems (a value out of bounds, a disallowed category, a missing
  value, a duplicate id) are handled per `strategy`:
    strict     -> raise DataValidationError listing every violation
    warn       -> log and keep all rows
    quarantine -> split into (valid_rows, invalid_rows) and let the caller
                  decide what to do with each
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from bnpl_credit_risk.constants import VALIDATION_QUARANTINE, VALIDATION_STRICT, VALIDATION_WARN
from bnpl_credit_risk.data.schemas import DataSchema
from bnpl_credit_risk.exceptions import DataValidationError


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    invalid: pd.DataFrame
    violations: list[str] = field(default_factory=list)
    row_issue_counts: dict[str, int] = field(default_factory=dict)

    @property
    def report(self) -> dict[str, Any]:
        return {
            "n_valid_rows": int(len(self.valid)),
            "n_invalid_rows": int(len(self.invalid)),
            "structural_violations": list(self.violations),
            "row_issue_counts": dict(self.row_issue_counts),
        }


class BNPLDataValidator:
    """Validates a dataframe against a `DataSchema` under a configurable policy."""

    def __init__(self, schema: DataSchema, strategy: str) -> None:
        self._schema = schema
        self._strategy = strategy

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        log = logger.bind(pipeline="data.validation")
        structural_violations = self._check_structure(df)
        if structural_violations:
            message = "Structural data contract violations: " + "; ".join(structural_violations)
            log.error(message)
            raise DataValidationError(message, violations=structural_violations)

        row_invalid_mask = pd.Series(False, index=df.index)
        issue_counts: dict[str, int] = {}

        def _flag(mask: pd.Series, issue_name: str) -> None:
            nonlocal row_invalid_mask
            n = int(mask.sum())
            if n:
                issue_counts[issue_name] = issue_counts.get(issue_name, 0) + n
                row_invalid_mask = row_invalid_mask | mask

        # Missing values in required columns
        for col in self._schema.required_columns:
            if col in df.columns:
                _flag(df[col].isnull(), f"missing_value:{col}")

        # Duplicate id_column
        if self._schema.id_column in df.columns:
            dup_mask = df[self._schema.id_column].duplicated(keep="first")
            _flag(dup_mask, f"duplicate_id:{self._schema.id_column}")

        # Numeric bounds
        for col, bound in self._schema.bounds.items():
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                out_of_bounds = numeric.notna() & ((numeric < bound.min) | (numeric > bound.max))
                _flag(out_of_bounds, f"out_of_bounds:{col}")

        # Allowed categorical values
        for col, allowed in self._schema.allowed_values.items():
            if col in df.columns:
                not_allowed = df[col].notna() & ~df[col].astype(str).isin(allowed)
                _flag(not_allowed, f"disallowed_value:{col}")

        # Target must be binary (training schema only)
        if self._schema.target_column and self._schema.target_column in df.columns:
            target = df[self._schema.target_column]
            not_binary = target.notna() & ~target.isin([0, 1])
            _flag(not_binary, f"non_binary_target:{self._schema.target_column}")

        n_invalid = int(row_invalid_mask.sum())
        if n_invalid:
            log.warning(
                "Row-level validation issues: {} / {} rows flagged — {}",
                n_invalid,
                len(df),
                issue_counts,
            )

        if n_invalid and self._strategy == VALIDATION_STRICT:
            violations = [f"{issue}={count}" for issue, count in issue_counts.items()]
            message = f"{n_invalid} row(s) violate the data contract under strict policy"
            raise DataValidationError(message, violations=violations)

        if self._strategy == VALIDATION_QUARANTINE:
            valid_df = df.loc[~row_invalid_mask].copy()
            invalid_df = df.loc[row_invalid_mask].copy()
        else:
            # strict (no violations reached this point) or warn: keep everything
            valid_df = df.copy()
            invalid_df = df.iloc[0:0].copy()

        log.info(
            "Validation complete strategy={} valid_rows={} invalid_rows={}",
            self._strategy,
            len(valid_df),
            len(invalid_df),
        )
        return ValidationResult(valid=valid_df, invalid=invalid_df, row_issue_counts=issue_counts)

    def _check_structure(self, df: pd.DataFrame) -> list[str]:
        violations = []
        missing_columns = [c for c in self._schema.required_columns if c not in df.columns]
        if missing_columns:
            violations.append(f"missing required columns: {missing_columns}")

        if self._schema.date_column in df.columns:
            parsed = pd.to_datetime(df[self._schema.date_column], errors="coerce")
            if parsed.isna().all() and len(df) > 0:
                violations.append(f"date column '{self._schema.date_column}' is entirely unparseable")

        if self._schema.target_column is None and self._strategy == VALIDATION_WARN:
            # inference schema: nothing to check here — absence of target is expected.
            pass

        return violations
