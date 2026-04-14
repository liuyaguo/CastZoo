#!/bin/sh

set -eu

SCRIPT_PATH="$0"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="/data/ygliu/myprojects/CastFamily"
DATASET_PATH="${ROOT_DIR}/datasets/ETT-small/ETTh1.csv"
RUN_BASE="${ROOT_DIR}/CastZoo/tmp_runs/patchtst_etth1"
SPEC_PATH="${RUN_BASE}/spec_patchtst_etth1.json"
HASH_STORE="${RUN_BASE}/evaluator.sha256"

mkdir -p "${RUN_BASE}"

cat > "${SPEC_PATH}" <<JSON
{
  "model": "PatchTST",
  "dataset_path": "${DATASET_PATH}",
  "time_col": "date",
  "target_col": "OT",
  "phase": "forecast",
  "train_ratio": 0.6,
  "val_ratio": 0.2,
  "seq_len": 96,
  "pred_len": 24,
  "eval_split": "val",
  "seed": 42,
  "hyperparams": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "train_epochs": 1,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_ff": 256,
    "dropout": 0.1,
    "factor": 3,
    "activation": "gelu"
  }
}
JSON

cd "${ROOT_DIR}"
python -m CastZoo.runner \
  --spec "${SPEC_PATH}" \
  --out-dir "${RUN_BASE}/runs" \
  --evaluator-hash-store "${HASH_STORE}"
