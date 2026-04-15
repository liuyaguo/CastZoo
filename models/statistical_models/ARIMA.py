"""ARIMA / SARIMA wrapper based on statsmodels."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


class Model:
    """Forecast with statsmodels ARIMA using the official estimator API."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._fitted: Any | None = None
        self._order: tuple[int, int, int] = (1, 1, 1)
        self._seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._trend: str | None = None
        self._fit_kwargs: dict[str, Any] = {}

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "ARIMA requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        train_series = train_payload["data"][:, target_index].astype(np.float64)

        self._order = tuple(int(v) for v in hyperparams.get("order", (1, 1, 1)))
        self._seasonal_order = tuple(
            int(v) for v in hyperparams.get("seasonal_order", (0, 0, 0, 0))
        )
        self._trend = hyperparams.get("trend")

        model = ARIMA(
            train_series,
            order=self._order,
            seasonal_order=self._seasonal_order,
            trend=self._trend,
        )
        self._fit_kwargs = {
            key: hyperparams[key]
            for key in ("start_params", "transformed", "includes_fixed", "method", "method_kwargs", "gls", "gls_kwargs", "cov_type", "cov_kwds", "return_params", "low_memory")
            if key in hyperparams
        }
        self._fitted = model.fit(**self._fit_kwargs)

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("ARIMA model must be fit before predict")
        steps = len(eval_payload["data"][:, target_index])
        return np.asarray(self._fitted.forecast(steps=steps), dtype=np.float64)

    def predict_windows(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
        """Forecast the eval split with rolling seq_len -> pred_len windows."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "ARIMA requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        order = tuple(int(v) for v in hyperparams.get("order", (1, 1, 1)))
        seasonal_order = tuple(
            int(v) for v in hyperparams.get("seasonal_order", (0, 0, 0, 0))
        )
        trend = hyperparams.get("trend")
        fit_kwargs = {
            key: hyperparams[key]
            for key in (
                "start_params",
                "transformed",
                "includes_fixed",
                "method",
                "method_kwargs",
                "gls",
                "gls_kwargs",
                "cov_type",
                "cov_kwds",
                "return_params",
                "low_memory",
            )
            if key in hyperparams
        }
        seq_len = int(spec["seq_len"])
        pred_len = int(spec["pred_len"])
        stride = int(hyperparams.get("window_stride", 1))

        series = eval_payload["data"][:, target_index].astype(np.float64)
        timestamps = eval_payload["timestamps"]

        available = len(series) - seq_len - pred_len + 1
        if available <= 0:
            raise ValueError(
                "Evaluation split is too short for ARIMA windowed forecasting "
                f"(seq_len={seq_len}, pred_len={pred_len})"
            )

        rows: list[dict[str, float | str]] = []
        actual_values: list[float] = []
        predicted_values: list[float] = []

        for start in range(0, available, stride):
            context = series[start : start + seq_len]
            actual = series[start + seq_len : start + seq_len + pred_len]
            target_timestamps = timestamps[start + seq_len : start + seq_len + pred_len]

            fitted = ARIMA(
                context,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
            ).fit(**fit_kwargs)
            predictions = np.asarray(fitted.forecast(steps=pred_len), dtype=np.float64)

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
