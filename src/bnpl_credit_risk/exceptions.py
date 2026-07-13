"""Business exceptions for the bnpl_credit_risk package.

Using specific exception types (rather than bare Exception/ValueError) lets
callers — CLI commands, pipelines, tests — distinguish a bad config from a
data contract violation from a missing artifact, and react accordingly
(e.g. the CLI maps each to a distinct process exit code).
"""

from __future__ import annotations


class BNPLError(Exception):
    """Base class for all errors raised by bnpl_credit_risk."""


class ConfigError(BNPLError):
    """A configuration file is missing, malformed, or internally inconsistent."""


class DataValidationError(BNPLError):
    """The input data violates the data contract under a `strict` validation policy."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []


class SchemaError(BNPLError):
    """The dataframe schema does not match what a component expects (columns/dtypes)."""


class FeatureEngineeringError(BNPLError):
    """A feature transformer could not be applied to the given input."""


class ModelNotFoundError(BNPLError):
    """No trained model artifact could be resolved for the requested version."""


class ArtifactNotFoundError(BNPLError):
    """A required artifact file (preprocessor, metadata, threshold, ...) is missing."""


class InferenceError(BNPLError):
    """Batch or realtime inference could not produce predictions for the given input."""
