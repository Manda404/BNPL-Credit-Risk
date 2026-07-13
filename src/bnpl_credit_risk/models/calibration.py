"""Probability calibration — wraps the full [feature engineering -> encoding ->
model] pipeline in `CalibratedClassifierCV` so predicted probabilities are
directly usable for risk banding and cost-based decisions, not just ranking.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from bnpl_credit_risk.settings import CalibrationConfig


class ProbabilityCalibrator:
    def __init__(self, calibration_config: CalibrationConfig) -> None:
        self._config = calibration_config

    def calibrate(
        self, uncalibrated_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
    ) -> CalibratedClassifierCV:
        log = logger.bind(pipeline="models.calibration")
        log.info(
            "Calibrating probabilities method={} cv={}", self._config.method, self._config.cv
        )
        calibrated = CalibratedClassifierCV(
            estimator=uncalibrated_pipeline, method=self._config.method, cv=self._config.cv
        )
        calibrated.fit(X_train, y_train)
        return calibrated
