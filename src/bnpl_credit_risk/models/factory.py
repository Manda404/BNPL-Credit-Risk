"""ModelFactory — a small registry so a new algorithm (CatBoost, LightGBM,
Logistic Regression) can be added later without touching the trainer. Only
`xgboost` is actually wired up today; the others are documented extension
points, not stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xgboost import XGBClassifier

from bnpl_credit_risk.exceptions import ConfigError

_ModelConstructor = Callable[[dict[str, Any]], Any]


def _build_xgboost(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(**params)


class ModelFactory:
    _registry: dict[str, _ModelConstructor] = {"xgboost": _build_xgboost}

    @classmethod
    def register(cls, name: str, constructor: _ModelConstructor) -> None:
        cls._registry[name] = constructor

    @classmethod
    def create(cls, algorithm: str, params: dict[str, Any]) -> Any:
        if algorithm not in cls._registry:
            raise ConfigError(
                f"Unknown algorithm '{algorithm}'. Registered: {list(cls._registry)}"
            )
        return cls._registry[algorithm](dict(params))
