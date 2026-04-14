from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CastZoo historical pattern similarity search")
    parser.add_argument("--dataset-path", required=True, help="Path to the source CSV dataset")
    parser.add_argument("--target-col", required=True, help="Name of the target column")
    parser.add_argument("--time-col", required=True, help="Name of the time/date column")
    parser.add_argument(
        "--query-window-start",
        required=True,
        type=int,
        help="Start index of the query window in the dataset",
    )
    parser.add_argument(
        "--query-window-end",
        required=True,
        type=int,
        help="End index of the query window (exclusive)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of most similar historical segments to return")
    return parser.parse_args(argv)


def pearson_sliding_similarity(series: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return Pearson correlation coefficients for each valid series window."""
    n = len(query)
    if n < 2 or len(series) < n:
        return np.array([], dtype=np.float64)

    query_std = float(query.std())
    if query_std < 1e-8:
        return np.zeros(len(series) - n + 1, dtype=np.float64)

    q_norm = (query - query.mean()) / (query_std + 1e-8)
    corr = correlate(series - series.mean(), q_norm, mode="valid")

    num_windows = len(series) - n + 1
    windows_std = np.array([series[index : index + n].std() for index in range(num_windows)], dtype=np.float64)
    windows_std = np.where(windows_std < 1e-8, 1e-8, windows_std)

    similarities = corr / (n * windows_std)
    return np.clip(similarities, -1.0, 1.0)


def build_matches(
    similarities: np.ndarray,
    time_values: np.ndarray,
    query_length: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or similarities.size == 0:
        return []

    k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-k:][::-1]

    matches: list[dict[str, Any]] = []
    for raw_index in top_indices:
        idx = int(raw_index)
        window_end = idx + query_length
        matches.append(
            {
                "end_index": window_end,
                "end_time": str(time_values[min(window_end - 1, len(time_values) - 1)]),
                "similarity": round(float(similarities[idx]), 4),
                "start_index": idx,
                "start_time": str(time_values[idx]),
            }
        )
    return matches


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.top_k <= 0:
        print(json.dumps({"error": "top_k must be positive", "matches": []}, sort_keys=True))
        sys.exit(1)

    try:
        df = pd.read_csv(args.dataset_path)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to read dataset: {exc}", "matches": []}, sort_keys=True))
        sys.exit(1)

    if args.target_col not in df.columns:
        print(
            json.dumps(
                {"error": f"Target column '{args.target_col}' not found in dataset", "matches": []},
                sort_keys=True,
            )
        )
        sys.exit(1)

    try:
        series = df[args.target_col].astype(float).to_numpy(copy=True)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to parse target column as float: {exc}", "matches": []}, sort_keys=True))
        sys.exit(1)

    if args.time_col in df.columns:
        time_values = df[args.time_col].to_numpy(copy=True)
    else:
        time_values = np.arange(len(series))

    start_idx = args.query_window_start
    end_idx = args.query_window_end
    if start_idx < 0 or end_idx > len(series) or start_idx >= end_idx:
        print(json.dumps({"error": f"Invalid query window [{start_idx}, {end_idx})", "matches": []}, sort_keys=True))
        sys.exit(1)

    query = series[start_idx:end_idx]
    train_series = series[:start_idx]
    if len(train_series) < len(query):
        print(json.dumps({"error": "Training series shorter than query window", "matches": []}, sort_keys=True))
        sys.exit(1)

    similarities = pearson_sliding_similarity(train_series, query)
    matches = build_matches(similarities, time_values, len(query), args.top_k)

    print(
        json.dumps(
            {
                "matches": matches,
                "query_length": int(len(query)),
                "search_length": int(len(train_series)),
                "top_k": min(args.top_k, len(similarities)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
