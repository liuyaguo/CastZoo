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
