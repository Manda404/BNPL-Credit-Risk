"""Predictor — the reusable scoring core.

Deliberately IO-agnostic: it takes a dataframe in and returns predictions
out. `inference/batch.py` wraps it with file reading/writing and validation
for scheduled batch runs; a future FastAPI service would wrap this same
class for realtime single-record scoring (see inference/realtime.py) instead
of reimplementing scoring logic.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bnpl_credit_risk.evaluation.thresholding import assign_risk_band
from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.models.registry import resolve_model_dir
from bnpl_credit_risk.settings import RiskBandConfig


class Predictor:
    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, models_dir: Path, model_version: str = "latest") -> Predictor:
        version_dir = resolve_model_dir(models_dir, model_version)
        return cls(ArtifactBundle.load(version_dir))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.bundle.pipeline.predict_proba(df))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "Published model predict_proba must return two class-probability columns"
            )
        return probabilities[:, 1]

    def score(self, df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        threshold = self.bundle.threshold["threshold"]
        risk_bands = [RiskBandConfig.model_validate(rb) for rb in self.bundle.threshold.get("risk_bands", [])]

        probabilities = self.predict_proba(df)
        predicted = (probabilities >= threshold).astype(int)
        scoring_timestamp = datetime.now().isoformat()

        return pd.DataFrame(
            {
                id_column: df[id_column].to_numpy(),
                "default_probability": probabilities,
                "predicted_default": predicted,
                "decision_threshold": threshold,
                "risk_band": [assign_risk_band(p, risk_bands) for p in probabilities],
                "model_version": self.bundle.version,
                "scoring_timestamp": scoring_timestamp,
            }
        )
