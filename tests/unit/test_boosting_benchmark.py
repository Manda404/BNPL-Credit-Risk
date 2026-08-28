from __future__ import annotations

import numpy as np
import pandas as pd

from bnpl_credit_risk.models.boosting import (
    CategoricalFrameAdapter,
    FittedBoostingModel,
)


class _ProbabilityEstimator:
    def predict_proba(self, frame):
        probability = np.full(len(frame), 0.3)
        return np.column_stack((1 - probability, probability))


def test_categorical_adapter_learns_vocabulary_from_training_only():
    train = pd.DataFrame({"segment": ["A", "B", None], "amount": [1, 2, 3]})
    validation = pd.DataFrame({"segment": ["A", "UNSEEN"], "amount": [4, 5]})

    adapter = CategoricalFrameAdapter.fit(train, ["segment"])
    transformed = adapter.transform_native(validation)

    assert transformed["segment"].cat.categories.tolist() == ["A", "B"]
    assert transformed["segment"].isna().tolist() == [False, True]


def test_catboost_adapter_replaces_missing_categories_with_explicit_token():
    frame = pd.DataFrame({"segment": ["A", None], "amount": [1, 2]})
    adapter = CategoricalFrameAdapter.fit(frame, ["segment"])

    transformed = adapter.transform_catboost(frame)

    assert transformed["segment"].tolist() == ["A", "__MISSING__"]


def test_fitted_boosting_model_exposes_sklearn_probability_contract():
    frame = pd.DataFrame({"amount": [10.0, 20.0]})
    fitted = FittedBoostingModel(
        model_name="LightGBM",
        estimator=_ProbabilityEstimator(),
        adapter=CategoricalFrameAdapter.fit(frame, []),
        feature_names=["amount"],
    )

    probabilities = fitted.predict_proba(frame)

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(fitted.predict_default_probability(frame), 0.3)
