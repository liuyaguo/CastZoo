"""Theta wrapper based on statsmodels ThetaModel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class Model:
    """Forecast with statsmodels ThetaModel."""

    def __init__(self, configs: SimpleNamespace):
        self.configs = configs
        self._fitted: Any | None = None

    def fit(self, train_payload: dict[str, np.ndarray], target_index: int, spec: dict[str, Any]) -> None:
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "Theta requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        hyperparams = dict(spec.get("hyperparams", {}))
        train_series = train_payload["data"][:, target_index].astype(np.float64)
        timestamps = pd.to_datetime(train_payload["timestamps"])
        freq = pd.infer_freq(timestamps)
        index = pd.DatetimeIndex(timestamps, freq=freq)
        series = pd.Series(train_series, index=index)

        period = int(hyperparams.get("period", hyperparams.get("seasonal_periods", 1)))
        model = ThetaModel(
            series,
            period=period,
            deseasonalize=bool(hyperparams.get("deseasonalize", True)),
            use_test=bool(hyperparams.get("use_test", True)),
            method=hyperparams.get("method", "auto"),
            difference=bool(hyperparams.get("difference", False)),
        )
        self._fitted = model.fit(
            use_mle=bool(hyperparams.get("use_mle", False)),
            disp=bool(hyperparams.get("disp", False)),
        )

    def predict(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Theta model must be fit before predict")
        steps = len(eval_payload["data"][:, target_index])
        return np.asarray(self._fitted.forecast(steps=steps), dtype=np.float64)

    def predict_windows(
        self,
        eval_payload: dict[str, np.ndarray],
        target_index: int,
        spec: dict[str, Any],
        train_payload: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
        """Forecast eval split with rolling seq_len -> pred_len windows."""
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated
            raise ModuleNotFoundError(
                "Theta requires statsmodels. Install it with `pip install statsmodels`."
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
                "Evaluation split is too short for Theta windowed forecasting "
                f"(seq_len={seq_len}, pred_len={pred_len})"
            )

        period = int(hyperparams.get("period", hyperparams.get("seasonal_periods", 1)))
        rows: list[dict[str, float | str]] = []
        actual_values: list[float] = []
        predicted_values: list[float] = []

        for start in range(0, available, stride):
            context = series[start : start + seq_len]
            context_timestamps = pd.to_datetime(timestamps[start : start + seq_len])
            actual = series[start + seq_len : start + seq_len + pred_len]
            target_timestamps = timestamps[start + seq_len : start + seq_len + pred_len]

            inferred_freq = pd.infer_freq(context_timestamps)
            if inferred_freq is not None:
                index = pd.DatetimeIndex(context_timestamps, freq=inferred_freq)
            else:
                index = pd.DatetimeIndex(context_timestamps)
            context_series = pd.Series(context, index=index)

            model = ThetaModel(
                context_series,
                period=period,
                deseasonalize=bool(hyperparams.get("deseasonalize", True)),
                use_test=bool(hyperparams.get("use_test", True)),
                method=hyperparams.get("method", "auto"),
                difference=bool(hyperparams.get("difference", False)),
            )
            fitted = model.fit(
                use_mle=bool(hyperparams.get("use_mle", False)),
                disp=bool(hyperparams.get("disp", False)),
            )
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
