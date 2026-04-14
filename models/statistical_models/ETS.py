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
