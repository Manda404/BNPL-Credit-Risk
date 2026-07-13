"""Minimal realtime scoring entry point.

Batch is the priority (see inference/batch.py); this module exists so a
single application record can be scored synchronously — e.g. from a future
FastAPI endpoint — without duplicating the scoring logic in `Predictor`.
No web server is built here on purpose: wrap `score_single` in a route when
a realtime API is actually needed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bnpl_credit_risk.inference.predictor import Predictor


def score_single(predictor: Predictor, record: dict[str, Any], id_column: str) -> dict[str, Any]:
    """Score one application record (as a flat dict) and return one prediction record."""
    df = pd.DataFrame([record])
    result = predictor.score(df, id_column)
    return {str(k): v for k, v in result.iloc[0].to_dict().items()}
