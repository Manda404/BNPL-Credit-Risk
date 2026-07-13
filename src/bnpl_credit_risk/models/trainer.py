"""ModelTrainer — assembles [feature engineering -> encoding -> XGBoost] into
one `sklearn.Pipeline` and fits it on the train split only.

`scale_pos_weight` is computed from `y_train` alone (the notebook computed it
from the full `y` before splitting, which lets test-set class balance leak
into a training-time hyperparameter).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from bnpl_credit_risk.features.preprocessing import build_preprocessing_pipeline
from bnpl_credit_risk.models.factory import ModelFactory
from bnpl_credit_risk.settings import FeaturesConfig, ModelConfig


class ModelTrainer:
    def __init__(self, model_config: ModelConfig, features_config: FeaturesConfig) -> None:
        self._model_config = model_config
        self._features_config = features_config

    def compute_scale_pos_weight(self, y_train: pd.Series) -> float:
        n_negative = int((y_train == 0).sum())
        n_positive = int((y_train == 1).sum())
        if n_positive == 0:
            raise ValueError("y_train contains no positive class — cannot compute scale_pos_weight")
        return n_negative / n_positive

    def build_pipeline(self, scale_pos_weight: float, date_column: str = "transaction_date") -> Pipeline:
        preprocessing = build_preprocessing_pipeline(
            self._features_config, self._model_config.risk_scope, date_column=date_column
        )
        params = dict(self._model_config.xgboost_params)
        params["scale_pos_weight"] = scale_pos_weight
        model = ModelFactory.create(self._model_config.algorithm, params)
        return Pipeline([*preprocessing.steps, ("model", model)])

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, date_column: str = "transaction_date"
    ) -> Pipeline:
        log = logger.bind(pipeline="models.trainer")
        scale_pos_weight = self.compute_scale_pos_weight(y_train)
        log.info(
            "risk_scope={} algorithm={} scale_pos_weight={:.4f} (train-only) n_train={}",
            self._model_config.risk_scope,
            self._model_config.algorithm,
            scale_pos_weight,
            len(X_train),
        )
        pipeline = self.build_pipeline(scale_pos_weight, date_column=date_column)
        pipeline.fit(X_train, y_train)
        log.info("Model trained successfully")
        return pipeline
