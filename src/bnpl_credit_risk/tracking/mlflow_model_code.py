"""Models-from-code entry point loaded by MLflow, not imported by the package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


class BNPLCreditRiskModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.model = joblib.load(context.artifacts["fitted_model"])
        self.threshold = float(
            json.loads(Path(context.artifacts["threshold"]).read_text(encoding="utf-8"))[
                "threshold"
            ]
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        probability = self.model.predict_default_probability(model_input)
        return pd.DataFrame(
            {
                "default_probability": probability,
                "predicted_label": (probability >= self.threshold).astype(int),
            },
            index=model_input.index,
        )


mlflow.models.set_model(BNPLCreditRiskModel())
