"""Utilities for statistical and machine-learning time-series wrappers."""

from __future__ import annotations

import numpy as np


def resolve_lag(requested_lag: int, train_size: int) -> int:
    """Resolve a safe lag length for tabular forecasting."""
    if train_size < 2:
        raise ValueError("Training split must contain at least 2 rows for tabular forecasting")
    return max(1, min(int(requested_lag), train_size - 1))


def split_target_and_exog(
    data: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the target series and optional exogenous matrix."""
    target = data[:, target_index].astype(np.float64)
    if data.shape[1] == 1:
        return target, None
    exog = np.delete(data.astype(np.float64), target_index, axis=1)
    return target, exog


def build_lagged_features(
    data: np.ndarray,
    target_index: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised tabular dataset from lagged target values."""
    target, exog = split_target_and_exog(data, target_index)
    rows: list[np.ndarray] = []
    labels: list[float] = []

    for end in range(lag, len(target)):
        features = [target[end - lag : end]]
        if exog is not None:
            features.append(exog[end])
        rows.append(np.concatenate(features, axis=0))
        labels.append(float(target[end]))

    if not rows:
        raise ValueError("Not enough rows to build lagged features")

    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def recursive_regression_forecast(
    regressor: object,
    train_data: np.ndarray,
    eval_data: np.ndarray,
    target_index: int,
    lag: int,
) -> np.ndarray:
    """Forecast the evaluation target recursively with lagged features."""
    train_target, _ = split_target_and_exog(train_data, target_index)
    _, eval_exog = split_target_and_exog(eval_data, target_index)

    history = train_target.astype(np.float64).tolist()
    predictions: list[float] = []

    for step in range(len(eval_data)):
        features = [np.asarray(history[-lag:], dtype=np.float64)]
        if eval_exog is not None:
            features.append(eval_exog[step])
        x = np.concatenate(features, axis=0).reshape(1, -1)
        prediction = float(regressor.predict(x)[0])
        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions, dtype=np.float64)


def filter_estimator_kwargs(estimator: object, hyperparams: dict[str, object]) -> dict[str, object]:
    """Keep only keyword arguments accepted by an sklearn-style estimator."""
    get_params = getattr(estimator, "get_params", None)
    if get_params is None:
        return dict(hyperparams)
    valid_keys = set(get_params().keys())
    return {key: value for key, value in hyperparams.items() if key in valid_keys}
