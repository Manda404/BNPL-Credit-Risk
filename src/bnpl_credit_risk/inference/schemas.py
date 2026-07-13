"""Output contract for a scored row — shared by batch and (future) realtime inference."""

from __future__ import annotations

from pydantic import BaseModel


class PredictionRecord(BaseModel):
    user_id: int
    default_probability: float
    predicted_default: int
    decision_threshold: float
    risk_band: str
    model_version: str
    scoring_timestamp: str
