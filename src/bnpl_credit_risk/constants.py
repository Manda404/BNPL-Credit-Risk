"""Fixed vocabularies used across the package.

These are enumerations of valid *choices*, not business parameters — actual
values (thresholds, column lists, model params) live in configs/*.yaml and
must not be duplicated here.
"""

from __future__ import annotations

RISK_SCOPE_APPLICATION = "application_risk"
RISK_SCOPE_BEHAVIORAL = "behavioral_risk"
RISK_SCOPES = (RISK_SCOPE_APPLICATION, RISK_SCOPE_BEHAVIORAL)

SPLIT_STRATIFIED_RANDOM = "stratified_random"
SPLIT_GROUP_BY_USER = "group_by_user"
SPLIT_TIME_BASED = "time_based"
SPLIT_STRATEGIES = (SPLIT_STRATIFIED_RANDOM, SPLIT_GROUP_BY_USER, SPLIT_TIME_BASED)

VALIDATION_STRICT = "strict"
VALIDATION_WARN = "warn"
VALIDATION_QUARANTINE = "quarantine"
VALIDATION_STRATEGIES = (VALIDATION_STRICT, VALIDATION_WARN, VALIDATION_QUARANTINE)

THRESHOLD_FIXED = "fixed"
THRESHOLD_BEST_F1 = "best_f1"
THRESHOLD_MIN_RECALL = "min_recall"
THRESHOLD_MIN_PRECISION = "min_precision"
THRESHOLD_COST_MATRIX = "cost_matrix"
THRESHOLD_POLICIES = (
    THRESHOLD_FIXED,
    THRESHOLD_BEST_F1,
    THRESHOLD_MIN_RECALL,
    THRESHOLD_MIN_PRECISION,
    THRESHOLD_COST_MATRIX,
)

CALIBRATION_SIGMOID = "sigmoid"
CALIBRATION_ISOTONIC = "isotonic"
CALIBRATION_METHODS = (CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC)

MODEL_ARTIFACT_FILENAME = "pipeline.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"
THRESHOLD_FILENAME = "threshold.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
MODEL_CARD_FILENAME = "model_card.md"
LATEST_POINTER_FILENAME = "latest.json"

SCHEMA_VERSION = "1.0"
