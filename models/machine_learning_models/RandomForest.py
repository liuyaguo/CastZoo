"""Random Forest wrapper for lag-based time-series forecasting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from CastZoo.utils.classical_forecasting import (
    build_windowed_regression_features,
    direct_window_regression_forecast,
    filter_estimator_kwargs,
)


class Model:
    """Forecast with sklearn RandomForestRegressor on lagged features."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._model: Any | None = None
        self._seq_len: int | None = None
        self._pred_len: int | None = None

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "RandomForest requires scikit-learn. Install it with `pip install scikit-learn`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        self._seq_len = int(spec["seq_len"])
        self._pred_len = int(spec["pred_len"])
        stride = int(hyperparams.get("window_stride", 1))
        x_train, y_train = build_windowed_regression_features(
            train_payload["data"],
            target_index,
            self._seq_len,
            self._pred_len,
            stride=stride,
        )

        kwargs = filter_estimator_kwargs(RandomForestRegressor(), hyperparams)
        kwargs.setdefault("random_state", spec.get("seed", 42))
        model = RandomForestRegressor(**kwargs)
        model.fit(x_train, y_train)
        self._model = model

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self._model is None or self._seq_len is None or self._pred_len is None:
            raise RuntimeError("RandomForest model must be fit before predict")
        _, _, predictions = direct_window_regression_forecast(
            self._model,
            eval_payload,
            target_index,
            self._seq_len,
            self._pred_len,
            stride=int(spec.get("hyperparams", {}).get("window_stride", 1)),
        )
        return predictions

    def predict_windows(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray],
    ) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
        if self._model is None or self._seq_len is None or self._pred_len is None:
            raise RuntimeError("RandomForest model must be fit before predict_windows")
        return direct_window_regression_forecast(
            self._model,
            eval_payload,
            target_index,
            self._seq_len,
            self._pred_len,
            stride=int(spec.get("hyperparams", {}).get("window_stride", 1)),
        )
