"""Native categorical boosting benchmark used by notebooks 04 and 05.

The notebook owns orchestration only. Model construction, fold isolation,
native categorical preparation, early stopping and final refit live here so
the same behavior can be tested and reused outside Jupyter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import DMatrix, XGBClassifier

from bnpl_credit_risk.settings import ModelConfig

METRIC_NAMES = (
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1 score",
    "MCC",
    "Brier score",
    "Log loss",
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the common comparison metrics used for every boosting model."""
    prediction = (probability >= threshold).astype(int)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, probability)),
        "PR-AUC": float(average_precision_score(y_true, probability)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1 score": float(f1_score(y_true, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "Brier score": float(brier_score_loss(y_true, probability)),
        "Log loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


@dataclass
class CategoricalFrameAdapter:
    """Persist the categories learned from one training partition only."""

    categorical_features: list[str]
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, categorical_features: list[str]) -> CategoricalFrameAdapter:
        categories = {
            column: frame[column].astype("string").dropna().unique().tolist()
            for column in categorical_features
        }
        return cls(list(categorical_features), categories)

    def transform_native(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare pandas categorical columns for XGBoost and LightGBM."""
        prepared = frame.copy()
        for column in self.categorical_features:
            prepared[column] = pd.Categorical(
                prepared[column].astype("string"),
                categories=self.categories[column],
            )
        return prepared

    def transform_catboost(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare explicit strings for CatBoost's native category handling."""
        prepared = frame.copy()
        for column in self.categorical_features:
            prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
        return prepared

    def transform(self, frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
        if model_name == "CatBoost":
            return self.transform_catboost(frame)
        return self.transform_native(frame)


@dataclass
class LearningHistory:
    train: list[float]
    validation: list[float]
    metric_name: str


@dataclass
class BoostingModelResult:
    model_name: str
    oof_probability: np.ndarray
    fold_metrics: pd.DataFrame
    histories: list[LearningHistory]
    best_iterations: list[int]

    @property
    def median_best_iteration(self) -> int:
        return max(1, int(np.median(self.best_iterations)))


@dataclass
class ShapExplanation:
    values: pd.DataFrame
    base_values: np.ndarray
    features: pd.DataFrame


@dataclass
class FittedBoostingModel:
    """Selected fitted estimator together with its category vocabulary."""

    model_name: str
    estimator: Any
    adapter: CategoricalFrameAdapter
    feature_names: list[str]

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.adapter.transform(frame.loc[:, self.feature_names], self.model_name)

    def predict_default_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the one-dimensional probability of default."""
        return np.asarray(self.estimator.predict_proba(self.prepare(frame))[:, 1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Expose the sklearn two-column contract required by ``Predictor``."""
        default_probability = self.predict_default_probability(frame)
        return np.column_stack((1 - default_probability, default_probability))

    def feature_importances(self) -> pd.Series:
        values = np.asarray(self.estimator.feature_importances_, dtype=float)
        return pd.Series(values, index=self.feature_names, name="Importance").sort_values(
            ascending=False
        )

    def shap_values(self, frame: pd.DataFrame) -> ShapExplanation:
        """Compute native TreeSHAP contributions without an extra dependency."""
        raw_features = frame.loc[:, self.feature_names].copy()
        prepared = self.prepare(raw_features)

        if self.model_name == "XGBoost":
            matrix = DMatrix(prepared, enable_categorical=True)
            contributions = self.estimator.get_booster().predict(matrix, pred_contribs=True)
        elif self.model_name == "LightGBM":
            contributions = self.estimator.predict(prepared, pred_contrib=True)
        elif self.model_name == "CatBoost":
            pool = Pool(
                prepared,
                cat_features=self.adapter.categorical_features,
                feature_names=self.feature_names,
            )
            contributions = self.estimator.get_feature_importance(pool, type="ShapValues")
        else:  # pragma: no cover - protected by typed configuration
            raise ValueError(f"Unsupported boosting model: {self.model_name}")

        contributions = np.asarray(contributions, dtype=float)
        return ShapExplanation(
            values=pd.DataFrame(
                contributions[:, :-1],
                columns=self.feature_names,
                index=raw_features.index,
            ),
            base_values=contributions[:, -1],
            features=raw_features,
        )


@dataclass
class BoostingBenchmarkResult:
    models: dict[str, BoostingModelResult]
    comparison: pd.DataFrame
    best_model_name: str
    selected_model: FittedBoostingModel
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def best_result(self) -> BoostingModelResult:
        return self.models[self.best_model_name]


class BoostingBenchmark:
    """Compare configured boosting estimators with identical stratified folds."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        numeric_features: list[str],
        categorical_features: list[str],
        random_seed: int,
    ) -> None:
        self._config = model_config
        self._numeric_features = list(numeric_features)
        self._categorical_features = list(categorical_features)
        self._feature_names = [*numeric_features, *categorical_features]
        self._random_seed = random_seed

    def run(
        self, X: pd.DataFrame, y: pd.Series, *, observer: Any | None = None
    ) -> BoostingBenchmarkResult:
        X = X.loc[:, self._feature_names].reset_index(drop=True)
        y = y.astype(int).reset_index(drop=True)
        results = {}
        for name in self._config.benchmark.models:
            if observer is None:
                results[name] = self._cross_validate(name, X, y)
            else:
                with observer.model_run(name):
                    results[name] = self._cross_validate(name, X, y)
                    observer.log_model_result(results[name], y)
        comparison = self._comparison(results)
        best_model_name = str(comparison.index[0])
        selected_model = self._fit_final(
            best_model_name,
            X,
            y,
            results[best_model_name].median_best_iteration,
        )
        logger.bind(pipeline="models.boosting").info(
            "Selected boosting model={} metric={} score={:.4f}",
            best_model_name,
            self._config.benchmark.selection_metric,
            comparison.loc[
                best_model_name,
                f"{self._config.benchmark.selection_metric} mean",
            ],
        )
        benchmark_result = BoostingBenchmarkResult(
            models=results,
            comparison=comparison,
            best_model_name=best_model_name,
            selected_model=selected_model,
            feature_names=self._feature_names,
            numeric_features=self._numeric_features,
            categorical_features=self._categorical_features,
        )
        if observer is not None:
            observer.complete(benchmark_result)
        return benchmark_result

    def _build_model(self, model_name: str, final_iterations: int | None = None) -> Any:
        early_stopping_rounds = self._config.cross_validation.early_stopping_rounds
        if model_name == "XGBoost":
            params = dict(self._config.xgboost_params)
            params.update(enable_categorical=True, tree_method="hist")
            if final_iterations is None:
                params["early_stopping_rounds"] = early_stopping_rounds
            else:
                params["n_estimators"] = final_iterations
                params.pop("early_stopping_rounds", None)
            return XGBClassifier(**params)
        if model_name == "LightGBM":
            params = dict(self._config.benchmark.lightgbm_params)
            params["random_state"] = self._random_seed
            if final_iterations is not None:
                params["n_estimators"] = final_iterations
            return LGBMClassifier(**params)
        if model_name == "CatBoost":
            params = dict(self._config.benchmark.catboost_params)
            params["random_seed"] = self._random_seed
            if final_iterations is not None:
                params["iterations"] = final_iterations
            return CatBoostClassifier(**params)
        raise ValueError(f"Unknown boosting model: {model_name}")

    def _cross_validate(
        self, model_name: str, X: pd.DataFrame, y: pd.Series
    ) -> BoostingModelResult:
        cv_config = self._config.cross_validation
        cv = StratifiedKFold(
            n_splits=cv_config.n_splits,
            shuffle=cv_config.shuffle,
            random_state=self._random_seed if cv_config.shuffle else None,
        )
        oof_probability = np.full(len(X), np.nan, dtype=float)
        fold_rows: list[dict[str, float | int]] = []
        histories: list[LearningHistory] = []
        best_iterations: list[int] = []

        for fold, (train_indices, validation_indices) in enumerate(cv.split(X, y), start=1):
            probability, history, best_iteration = self._fit_fold(
                model_name,
                X.iloc[train_indices],
                y.iloc[train_indices],
                X.iloc[validation_indices],
                y.iloc[validation_indices],
            )
            oof_probability[validation_indices] = probability
            fold_rows.append(
                {
                    "Fold": fold,
                    **classification_metrics(
                        y.iloc[validation_indices],
                        probability,
                        self._config.benchmark.comparison_threshold,
                    ),
                }
            )
            histories.append(history)
            best_iterations.append(best_iteration)
            logger.bind(pipeline="models.boosting").info(
                "{} fold {}/{} ROC-AUC={:.4f} best_iteration={}",
                model_name,
                fold,
                cv.n_splits,
                fold_rows[-1]["ROC-AUC"],
                best_iteration,
            )

        if np.isnan(oof_probability).any():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {model_name}")
        return BoostingModelResult(
            model_name=model_name,
            oof_probability=oof_probability,
            fold_metrics=pd.DataFrame(fold_rows).set_index("Fold"),
            histories=histories,
            best_iterations=best_iterations,
        )

    def _fit_fold(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> tuple[np.ndarray, LearningHistory, int]:
        adapter = CategoricalFrameAdapter.fit(X_train, self._categorical_features)
        train_input = adapter.transform(X_train, model_name)
        validation_input = adapter.transform(X_validation, model_name)
        model = self._build_model(model_name)
        early_stopping_rounds = self._config.cross_validation.early_stopping_rounds

        if model_name == "CatBoost":
            model.fit(
                train_input,
                y_train,
                eval_set=(validation_input, y_validation),
                cat_features=self._categorical_features,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
            raw_history = model.get_evals_result()
            train_key, validation_key = "learn", "validation"
            best_iteration = max(1, model.get_best_iteration() + 1)
        elif model_name == "XGBoost":
            model.fit(
                train_input,
                y_train,
                eval_set=[
                    (train_input, y_train),
                    (validation_input, y_validation),
                ],
                verbose=False,
            )
            raw_history = model.evals_result()
            train_key, validation_key = "validation_0", "validation_1"
            best_iteration = max(1, model.best_iteration + 1)
        else:
            model.fit(
                train_input,
                y_train,
                eval_X=validation_input,
                eval_y=y_validation,
                eval_names=["validation"],
                callbacks=[
                    log_evaluation(period=0),
                    early_stopping(early_stopping_rounds, verbose=False),
                ],
            )
            validation_history = model.evals_result_
            metric_name = next(iter(validation_history["validation"]))
            best_iteration = max(1, model.best_iteration_)
            validation_curve = validation_history["validation"][metric_name][:best_iteration]
            train_curve = [
                log_loss(
                    y_train,
                    model.predict_proba(train_input, num_iteration=iteration)[:, 1],
                    labels=[0, 1],
                )
                for iteration in range(1, best_iteration + 1)
            ]
            raw_history = {
                "train": {metric_name: train_curve},
                "validation": {metric_name: validation_curve},
            }
            train_key, validation_key = "train", "validation"

        metric_name = next(iter(raw_history[train_key]))
        history = LearningHistory(
            train=list(raw_history[train_key][metric_name]),
            validation=list(raw_history[validation_key][metric_name]),
            metric_name=metric_name,
        )
        probability = np.asarray(model.predict_proba(validation_input)[:, 1])
        return probability, history, best_iteration

    def _comparison(self, results: dict[str, BoostingModelResult]) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for model_name, result in results.items():
            summary = result.fold_metrics.agg(["mean", "std"])
            row: dict[str, float | int | str] = {"Model": model_name}
            for metric in METRIC_NAMES:
                row[f"{metric} mean"] = float(summary.loc["mean", metric])
                row[f"{metric} std"] = float(summary.loc["std", metric])
            row["Median best iteration"] = result.median_best_iteration
            rows.append(row)
        selection = self._config.benchmark.selection_metric
        secondary = [metric for metric in ("ROC-AUC", "PR-AUC", "MCC") if metric != selection]
        return (
            pd.DataFrame(rows)
            .set_index("Model")
            .sort_values(
                [f"{selection} mean", *(f"{metric} mean" for metric in secondary)],
                ascending=False,
            )
        )

    def _fit_final(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        iterations: int,
    ) -> FittedBoostingModel:
        adapter = CategoricalFrameAdapter.fit(X, self._categorical_features)
        train_input = adapter.transform(X, model_name)
        model = self._build_model(model_name, final_iterations=iterations)
        if model_name == "CatBoost":
            model.fit(
                train_input,
                y,
                cat_features=self._categorical_features,
                verbose=False,
            )
        elif model_name == "LightGBM":
            model.fit(train_input, y, callbacks=[log_evaluation(period=0)])
        else:
            model.fit(train_input, y, verbose=False)
        return FittedBoostingModel(
            model_name=model_name,
            estimator=model,
            adapter=adapter,
            feature_names=self._feature_names,
        )
