"""Prophet wrapper based on the official Prophet API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class Model:
    """Forecast with Prophet using train/eval timestamps and optional regressors."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._model: Any | None = None
        self._extra_regressors: list[tuple[str, int]] = []

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from prophet import Prophet
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "Prophet requires `prophet`. Install it with `pip install prophet`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        feature_cols = _feature_columns(spec)

        train_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(train_payload["timestamps"]),
                "y": train_payload["data"][:, target_index].astype(np.float64),
            }
        )

        model = Prophet(
            growth=hyperparams.get("growth", "linear"),
            yearly_seasonality=hyperparams.get("yearly_seasonality", "auto"),
            weekly_seasonality=hyperparams.get("weekly_seasonality", "auto"),
            daily_seasonality=hyperparams.get("daily_seasonality", "auto"),
            seasonality_mode=hyperparams.get("seasonality_mode", "additive"),
            changepoint_prior_scale=hyperparams.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=hyperparams.get("seasonality_prior_scale", 10.0),
            holidays_prior_scale=hyperparams.get("holidays_prior_scale", 10.0),
            interval_width=hyperparams.get("interval_width", 0.8),
        )

        self._extra_regressors = []
        for index, name in enumerate(feature_cols):
            if index == target_index:
                continue
            model.add_regressor(name)
            train_df[name] = train_payload["data"][:, index].astype(np.float64)
            self._extra_regressors.append((name, index))

        model.fit(train_df)
        self._model = model

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Prophet model must be fit before predict")

        future = pd.DataFrame({"ds": pd.to_datetime(eval_payload["timestamps"])})
        for name, index in self._extra_regressors:
            future[name] = eval_payload["data"][:, index].astype(np.float64)

        forecast = self._model.predict(future)
        return forecast["yhat"].to_numpy(dtype=np.float64)


def _feature_columns(spec: dict[str, Any]) -> list[str]:
    header = pd.read_csv(spec["dataset_path"], nrows=0)
    return [column for column in header.columns if column != spec["time_col"]]
