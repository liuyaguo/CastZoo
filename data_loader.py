"""Data loading with frozen split enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def reject_if_test_in_forecast(spec: dict) -> None:
    """Reject forecast-phase specs that attempt to access the test split."""
    phase = spec.get("phase", "forecast")
    eval_split = spec.get("eval_split", "val")
    if phase == "forecast" and eval_split == "test":
        raise SystemExit(
            "SAFETY VIOLATION: Test split is inaccessible during forecast phase. "
            f"spec.phase={phase!r}, spec.eval_split={eval_split!r}"
        )


def compute_split_indices(n_rows: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Compute absolute split boundaries from the spec ratios."""
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return train_end, val_end


def load_dataset(spec: dict, allowed_splits: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load a CSV dataset and return only the requested split payloads."""
    if not allowed_splits:
        raise ValueError("allowed_splits must not be empty")

    df = pd.read_csv(spec["dataset_path"])
    time_col = spec["time_col"]
    feature_cols = [column for column in df.columns if column != time_col]
    feature_data = df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = df[time_col].to_numpy()

    train_end, val_end = compute_split_indices(
        len(df), spec["train_ratio"], spec["val_ratio"]
    )

    scaler = StandardScaler()
    scaler.fit(feature_data[:train_end])
    scaled_data = scaler.transform(feature_data)

    split_ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, len(df)),
    }

    result: dict[str, dict[str, np.ndarray]] = {}
    for split_name in allowed_splits:
        if split_name not in split_ranges:
            raise ValueError(f"Unknown split: {split_name!r}")
        start, end = split_ranges[split_name]
        result[split_name] = {
            "data": scaled_data[start:end],
            "raw": feature_data[start:end],
            "timestamps": timestamps[start:end],
        }

    return result
