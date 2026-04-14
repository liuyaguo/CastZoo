"""CatBoost wrapper for lag-based time-series forecasting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from CastZoo.utils.classical_forecasting import (
    build_lagged_features,
    filter_estimator_kwargs,
    recursive_regression_forecast,
    resolve_lag,
)


class Model:
    """Forecast with CatBoostRegressor on lagged features."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._model: Any | None = None
        self._lag: int | None = None

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from catboost import CatBoostRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "CatBoost requires `catboost`. Install it with `pip install catboost`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        self._lag = resolve_lag(int(spec["seq_len"]), len(train_payload["data"]))
        x_train, y_train = build_lagged_features(train_payload["data"], target_index, self._lag)

        kwargs = filter_estimator_kwargs(CatBoostRegressor(), hyperparams)
        kwargs.setdefault("loss_function", "RMSE")
        kwargs.setdefault("random_seed", spec.get("seed", 42))
        kwargs.setdefault("verbose", False)
        model = CatBoostRegressor(**kwargs)
        model.fit(x_train, y_train)
        self._model = model

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self._model is None or self._lag is None:
            raise RuntimeError("CatBoost model must be fit before predict")
        return recursive_regression_forecast(
            self._model,
            train_payload["data"],
            eval_payload["data"],
            target_index,
            self._lag,
        )
