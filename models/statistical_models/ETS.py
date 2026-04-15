"""ETS wrapper based on statsmodels ExponentialSmoothing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


class Model:
    """Forecast with Holt-Winters Exponential Smoothing."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._fitted: Any | None = None

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "ETS requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        train_series = train_payload["data"][:, target_index].astype(np.float64)

        model = ExponentialSmoothing(
            train_series,
            trend=hyperparams.get("trend"),
            damped_trend=bool(hyperparams.get("damped_trend", False)),
            seasonal=hyperparams.get("seasonal"),
            seasonal_periods=hyperparams.get("seasonal_periods"),
            initialization_method=hyperparams.get("initialization_method", "estimated"),
        )
        fit_kwargs = {
            key: hyperparams[key]
            for key in (
                "smoothing_level",
                "smoothing_trend",
                "smoothing_seasonal",
                "damping_trend",
                "optimized",
                "remove_bias",
                "start_params",
                "method",
                "minimize_kwargs",
                "use_brute",
                "use_boxcox",
                "use_basinhopping",
                "initial_level",
                "initial_trend",
            )
            if key in hyperparams
        }
        self._fitted = model.fit(**fit_kwargs)

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("ETS model must be fit before predict")
        steps = len(eval_payload["data"][:, target_index])
        return np.asarray(self._fitted.forecast(steps), dtype=np.float64)

    def predict_windows(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
        """Forecast eval split with rolling seq_len -> pred_len windows."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "ETS requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        seq_len = int(spec["seq_len"])
        pred_len = int(spec["pred_len"])
        stride = int(hyperparams.get("window_stride", 1))

        series = eval_payload["data"][:, target_index].astype(np.float64)
        timestamps = eval_payload["timestamps"]
        available = len(series) - seq_len - pred_len + 1
        if available <= 0:
            raise ValueError(
                "Evaluation split is too short for ETS windowed forecasting "
                f"(seq_len={seq_len}, pred_len={pred_len})"
            )

        fit_kwargs = {
            key: hyperparams[key]
            for key in (
                "smoothing_level",
                "smoothing_trend",
                "smoothing_seasonal",
                "damping_trend",
                "optimized",
                "remove_bias",
                "start_params",
                "method",
                "minimize_kwargs",
                "use_brute",
                "use_boxcox",
                "use_basinhopping",
                "initial_level",
                "initial_trend",
            )
            if key in hyperparams
        }

        rows: list[dict[str, float | str]] = []
        actual_values: list[float] = []
        predicted_values: list[float] = []

        for start in range(0, available, stride):
            context = series[start : start + seq_len]
            actual = series[start + seq_len : start + seq_len + pred_len]
            target_timestamps = timestamps[start + seq_len : start + seq_len + pred_len]

            model = ExponentialSmoothing(
                context,
                trend=hyperparams.get("trend"),
                damped_trend=bool(hyperparams.get("damped_trend", False)),
                seasonal=hyperparams.get("seasonal"),
                seasonal_periods=hyperparams.get("seasonal_periods"),
                initialization_method=hyperparams.get("initialization_method", "estimated"),
            )
            fitted = model.fit(**fit_kwargs)
            predictions = np.asarray(fitted.forecast(pred_len), dtype=np.float64)

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
