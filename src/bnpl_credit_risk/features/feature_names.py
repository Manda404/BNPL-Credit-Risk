"""Helpers to extract the final, ordered feature names produced by a fitted
preprocessing pipeline — needed to label feature importances and to persist
`feature_schema.json` alongside a trained model.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def get_output_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    column_transform = fitted_pipeline.named_steps["column_transform"]
    return list(column_transform.get_feature_names_out())


def get_feature_importances(pipeline: Any) -> pd.Series:
    """Feature importances for a fitted `[feature engineering -> encoding ->
    model]` pipeline, whether or not it's wrapped in `CalibratedClassifierCV`
    (calibration re-fits the estimator internally, so the pipeline with the
    actual `.feature_importances_` has to be unwrapped first).

    Returns an empty Series if importances can't be extracted — this is a
    reporting aid, not something that should ever fail training or inference.
    """
    try:
        base_pipeline = (
            pipeline.calibrated_classifiers_[0].estimator
            if hasattr(pipeline, "calibrated_classifiers_")
            else pipeline
        )
        feature_names = get_output_feature_names(base_pipeline)
        return pd.Series(base_pipeline.named_steps["model"].feature_importances_, index=feature_names)
    except Exception:  # pragma: no cover - defensive, importance is best-effort
        return pd.Series(dtype=float)
