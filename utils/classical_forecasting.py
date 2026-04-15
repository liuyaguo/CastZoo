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


def build_windowed_regression_features(
    data: np.ndarray,
    target_index: int,
    seq_len: int,
    pred_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build direct multi-step regression samples from seq_len history windows."""
    if seq_len < 1 or pred_len < 1:
        raise ValueError("seq_len and pred_len must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")

    available = len(data) - seq_len - pred_len + 1
    if available <= 0:
        raise ValueError("Not enough rows to build windowed regression features")

    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for start in range(0, available, stride):
        history = data[start : start + seq_len].astype(np.float64).reshape(-1)
        future = data[start + seq_len : start + seq_len + pred_len, target_index].astype(np.float64)
        rows.append(history)
        labels.append(future)

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


def direct_window_regression_forecast(
    regressor: object,
    eval_payload: dict[str, np.ndarray],
    target_index: int,
    seq_len: int,
    pred_len: int,
    stride: int = 1,
) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
    """Forecast an eval split with direct seq_len -> pred_len windows."""
    if seq_len < 1 or pred_len < 1:
        raise ValueError("seq_len and pred_len must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")

    data = eval_payload["data"]
    timestamps = eval_payload["timestamps"]
    available = len(data) - seq_len - pred_len + 1
    if available <= 0:
        raise ValueError(
            "Evaluation split is too short for windowed regression "
            f"(seq_len={seq_len}, pred_len={pred_len})"
        )

    rows: list[dict[str, float | str]] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []

    for start in range(0, available, stride):
        history = data[start : start + seq_len].astype(np.float64).reshape(1, -1)
        actual = data[start + seq_len : start + seq_len + pred_len, target_index].astype(np.float64)
        target_timestamps = timestamps[start + seq_len : start + seq_len + pred_len]

        raw_prediction = np.asarray(regressor.predict(history), dtype=np.float64)
        if raw_prediction.ndim == 2:
            prediction = raw_prediction[0]
        elif raw_prediction.ndim == 1:
            prediction = raw_prediction
        else:
            raise ValueError(
                f"Windowed regressor returned an unsupported prediction shape {raw_prediction.shape}"
            )

        if len(prediction) != pred_len:
            raise ValueError(
                f"Windowed regressor returned {len(prediction)} values for pred_len={pred_len}"
            )

        for timestamp, actual_value, predicted_value in zip(
            target_timestamps, actual, prediction, strict=True
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


def filter_estimator_kwargs(estimator: object, hyperparams: dict[str, object]) -> dict[str, object]:
    """Keep only keyword arguments accepted by an sklearn-style estimator."""
    get_params = getattr(estimator, "get_params", None)
    if get_params is None:
        return dict(hyperparams)
    valid_keys = set(get_params().keys())
    return {key: value for key, value in hyperparams.items() if key in valid_keys}
