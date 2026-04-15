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

    def predict_windows(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
        """Forecast eval split with rolling seq_len -> pred_len windows."""
        try:
            from prophet import Prophet
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "Prophet requires `prophet`. Install it with `pip install prophet`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        seq_len = int(spec["seq_len"])
        pred_len = int(spec["pred_len"])
        stride = int(hyperparams.get("window_stride", 1))

        data = eval_payload["data"].astype(np.float64)
        timestamps = eval_payload["timestamps"]
        series = data[:, target_index].astype(np.float64)
        available = len(series) - seq_len - pred_len + 1
        if available <= 0:
            raise ValueError(
                "Evaluation split is too short for Prophet windowed forecasting "
                f"(seq_len={seq_len}, pred_len={pred_len})"
            )

        feature_cols = _feature_columns(spec)
        exog_indices = [(name, idx) for idx, name in enumerate(feature_cols) if idx != target_index]

        rows: list[dict[str, float | str]] = []
        actual_values: list[float] = []
        predicted_values: list[float] = []

        for start in range(0, available, stride):
            context_slice = slice(start, start + seq_len)
            target_slice = slice(start + seq_len, start + seq_len + pred_len)

            context_timestamps = pd.to_datetime(timestamps[context_slice])
            target_timestamps = timestamps[target_slice]
            actual = series[target_slice]

            train_df = pd.DataFrame(
                {
                    "ds": context_timestamps,
                    "y": series[context_slice],
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

            for name, idx in exog_indices:
                model.add_regressor(name)
                train_df[name] = data[context_slice, idx]

            model.fit(train_df)

            future = pd.DataFrame({"ds": pd.to_datetime(timestamps[target_slice])})
            for name, idx in exog_indices:
                future[name] = data[target_slice, idx]
            forecast = model.predict(future)
            predictions = forecast["yhat"].to_numpy(dtype=np.float64)

            for timestamp, actual_value, predicted_value in zip(
                target_timestamps, actual, predictions, strict=True
            ):
                actual_values.append(float(actual_value))
                predicted_values.append(float(predicted_value))
                rows.append(
                    {
                        "timestamp": str(timestamp),
                        "actual": float(actual_value),
                        "predicted": float(predicted_value),
                    }
                )

        return (
            rows,
            np.asarray(actual_values, dtype=np.float64),
            np.asarray(predicted_values, dtype=np.float64),
        )


def _feature_columns(spec: dict[str, Any]) -> list[str]:
    header = pd.read_csv(spec["dataset_path"], nrows=0)
    return [column for column in header.columns if column != spec["time_col"]]
