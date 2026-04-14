#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/Users/liuyaguo/Desktop/CastFamily"
DATASET_PATH="${ROOT_DIR}/datasets/ETT-small/ETTh1.csv"
RUN_BASE="${ROOT_DIR}/CastZoo/tmp_runs/arima_etth1"
SPEC_PATH="${RUN_BASE}/spec_arima_etth1.json"
HASH_STORE="${RUN_BASE}/evaluator.sha256"

mkdir -p "${RUN_BASE}"

cat > "${SPEC_PATH}" <<'JSON'
{
  "model": "ARIMA",
  "dataset_path": "/Users/liuyaguo/Desktop/CastFamily/datasets/ETT-small/ETTh1.csv",
  "time_col": "date",
  "target_col": "OT",
  "phase": "post-forecast",
  "train_ratio": 0.6,
  "val_ratio": 0.2,
  "seq_len": 96,
  "pred_len": 96,
  "eval_split": "test",
  "hyperparams": {
    "order": [1, 1, 1],
    "window_stride": 1
  }
}
JSON

python -m CastZoo.runner \
  --spec "${SPEC_PATH}" \
  --out-dir "${RUN_BASE}/runs" \
  --evaluator-hash-store "${HASH_STORE}"
