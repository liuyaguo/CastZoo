# NOTE: This script analyzes the FULL dataset for statistical characterization.
# This is standard practice for pre-forecast analysis and does NOT violate the
# train/val/test split safety invariant (which applies to model training only).

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, ks_2samp
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


ANALYZER_VERSION = "1.0.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CastZoo quantitative dataset analyzer")
    parser.add_argument("--dataset-path", required=True, help="Path to a CSV dataset")
    parser.add_argument("--target-col", required=True, help="Target column name")
    parser.add_argument("--time-col", required=True, help="Time column name")
    parser.add_argument("--out", help="Optional JSON output path")
    return parser.parse_args(argv)


def _clean_numeric_series(series: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    return values[np.isfinite(values)]


def _stationarity_fallback(note: str) -> dict[str, Any]:
    return {
        "adf_statistic": None,
        "adf_pvalue": None,
        "adf_conclusion": "non-stationary",
        "kpss_statistic": None,
        "kpss_pvalue": None,
        "kpss_conclusion": "non-stationary",
        "overall": "uncertain",
        "kpss_note": note,
    }


def analyze_trend(series: np.ndarray) -> dict[str, float | str]:
    values = _clean_numeric_series(series)
    if len(values) < 2:
        return {"slope": 0.0, "r_squared": 0.0, "direction": "flat"}

    regression = linregress(np.arange(len(values), dtype=np.float64), values)
    slope = float(regression.slope)
    if slope > 0.001:
        direction = "upward"
    elif slope < -0.001:
        direction = "downward"
    else:
        direction = "flat"

    return {
        "slope": slope,
        "r_squared": float(regression.rvalue**2),
        "direction": direction,
    }


def analyze_seasonality(series: np.ndarray) -> dict[str, int | float | str]:
    values = _clean_numeric_series(series)
    max_lag = min(len(values) // 2, 1000)
    if len(values) < 4 or max_lag < 2:
        return {"dominant_period": 0, "strength": 0.0, "method": "autocorrelation"}

    centered = values - np.mean(values)
    if np.allclose(centered, 0.0):
        return {"dominant_period": 0, "strength": 0.0, "method": "autocorrelation"}

    scores: list[tuple[int, float]] = []
    for lag in range(2, max_lag + 1):
        lead = centered[:-lag]
        lagged = centered[lag:]
        denominator = np.sqrt(np.dot(lead, lead) * np.dot(lagged, lagged))
        if denominator == 0:
            continue
        autocorrelation = float(np.dot(lead, lagged) / denominator)
        if autocorrelation > 0:
            scores.append((lag, autocorrelation))

    if not scores:
        return {"dominant_period": 0, "strength": 0.0, "method": "autocorrelation"}

    best_strength = max(strength for _, strength in scores)
    best_lag = min(
        lag
        for lag, strength in scores
        if np.isclose(strength, best_strength, atol=1e-9, rtol=1e-6)
    )

    return {
        "dominant_period": int(best_lag),
        "strength": float(min(max(best_strength, 0.0), 1.0)),
        "method": "autocorrelation",
    }


def analyze_stationarity(series: np.ndarray) -> dict[str, Any]:
    values = _clean_numeric_series(series)
    if len(values) < 8 or np.allclose(values, values[0]):
        return _stationarity_fallback("Series is too short or constant for reliable ADF/KPSS tests.")

    try:
        adf_statistic, adf_pvalue, *_ = adfuller(values, autolag="AIC")
    except ValueError as error:
        return _stationarity_fallback(f"ADF failed: {error}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_statistic, kpss_pvalue, *_ = kpss(values, regression="c", nlags="auto")
    except ValueError as error:
        return _stationarity_fallback(f"KPSS failed: {error}")

    adf_conclusion = "stationary" if adf_pvalue < 0.05 else "non-stationary"
    kpss_conclusion = "stationary" if kpss_pvalue > 0.05 else "non-stationary"
    if adf_conclusion == "stationary" and kpss_conclusion == "stationary":
        overall = "stationary"
    elif adf_conclusion == "non-stationary" and kpss_conclusion == "non-stationary":
        overall = "non-stationary"
    else:
        overall = "uncertain"

    result: dict[str, Any] = {
        "adf_statistic": float(adf_statistic),
        "adf_pvalue": float(adf_pvalue),
        "adf_conclusion": adf_conclusion,
        "kpss_statistic": float(kpss_statistic),
        "kpss_pvalue": float(kpss_pvalue),
        "kpss_conclusion": kpss_conclusion,
        "overall": overall,
    }
    if np.isclose(kpss_pvalue, 0.01) or np.isclose(kpss_pvalue, 0.1):
        result["kpss_note"] = (
            "KPSS p-value hit the statsmodels interpolation boundary; reported numeric value is capped."
        )
    return result


def analyze_volatility(series: np.ndarray) -> dict[str, float]:
    values = _clean_numeric_series(series)
    if len(values) < 2:
        return {"std": 0.0, "cv": 0.0, "rolling_std_mean": 0.0}

    std = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    cv = float(std / abs(mean)) if mean != 0 else float("inf")

    window = min(max(len(values) // 20, 10), len(values))
    rolling = pd.Series(values).rolling(window=window, min_periods=2).std(ddof=1).dropna()
    rolling_std_mean = float(rolling.mean()) if not rolling.empty else 0.0

    return {"std": std, "cv": cv, "rolling_std_mean": rolling_std_mean}


def analyze_missing_data(df: pd.DataFrame, target_col: str) -> dict[str, int | float]:
    is_missing = df[target_col].isna().to_numpy()
    count = int(is_missing.sum())
    rate = float(count / len(df)) if len(df) else 0.0

    longest_gap = 0
    current_gap = 0
    for value in is_missing:
        if value:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
        else:
            current_gap = 0

    return {"count": count, "rate": rate, "longest_gap": int(longest_gap)}


def analyze_anomaly_density(series: np.ndarray) -> dict[str, int | float | str]:
    values = _clean_numeric_series(series)
    if len(values) == 0:
        return {"count": 0, "rate": 0.0, "method": "iqr_3sigma"}

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    anomalies = (values < lower) | (values > upper)
    count = int(np.count_nonzero(anomalies))

    return {
        "count": count,
        "rate": float(count / len(values)),
        "method": "iqr_3sigma",
    }


def analyze_distribution_shifts(series: np.ndarray, n_windows: int = 10) -> dict[str, Any]:
    values = _clean_numeric_series(series)
    if len(values) < 4:
        return {
            "detected": False,
            "n_changepoints": 0,
            "method": "ks_test_windows",
            "changepoint_indices": [],
        }

    window_count = min(max(n_windows, 2), len(values))
    windows = np.array_split(values, window_count)
    changepoint_indices: list[int] = []
    boundary = 0
    for left, right in zip(windows, windows[1:]):
        boundary += len(left)
        if len(left) == 0 or len(right) == 0:
            continue
        _, pvalue = ks_2samp(left, right)
        if pvalue < 0.01:
            changepoint_indices.append(int(boundary))

    return {
        "detected": bool(changepoint_indices),
        "n_changepoints": len(changepoint_indices),
        "method": "ks_test_windows",
        "changepoint_indices": changepoint_indices,
    }


def analyze(
    dataset_path: str | Path,
    target_col: str,
    time_col: str,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)

    if time_col not in df.columns:
        raise ValueError(f"Time column {time_col!r} not found in dataset")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in dataset")

    ordered = df.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    ordered = ordered.sort_values(time_col).reset_index(drop=True)
    ordered[target_col] = pd.to_numeric(ordered[target_col], errors="coerce")

    numeric_series = ordered[target_col].to_numpy(dtype=np.float64, copy=True)
    clean_series = _clean_numeric_series(numeric_series)
    feature_columns = [column for column in ordered.columns if column != time_col]

    result = {
        "dataset_path": str(dataset_path),
        "target_col": target_col,
        "time_col": time_col,
        "n_rows": int(len(ordered)),
        "n_features": int(len(feature_columns)),
        "analysis": {
            "trend": analyze_trend(clean_series),
            "seasonality": analyze_seasonality(clean_series),
            "stationarity": analyze_stationarity(clean_series),
            "volatility": analyze_volatility(clean_series),
            "missing_data": analyze_missing_data(ordered, target_col),
            "anomaly_density": analyze_anomaly_density(clean_series),
            "distribution_shifts": analyze_distribution_shifts(clean_series),
        },
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "analyzer_version": ANALYZER_VERSION,
    }

    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze(
        dataset_path=args.dataset_path,
        target_col=args.target_col,
        time_col=args.time_col,
        out_path=args.out,
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
