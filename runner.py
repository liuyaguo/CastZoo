"""Unified experiment runner for CastZoo experiments."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time as _time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from CastZoo.data_loader import load_dataset, reject_if_test_in_forecast
from CastZoo.evaluator import evaluate
from CastZoo.utils.device import detect_device, get_device_info
from CastZoo.utils.hash import compute_spec_hash
from CastZoo.utils.seed import set_seed


class LazyModelDict(dict):
    """Lazy-load models from the copied CastZoo.models package."""

    def __init__(self, models_dir: str) -> None:
        models_root = Path(models_dir)
        self._map = {
            path.stem: ".".join(("CastZoo",) + path.relative_to(models_root.parent).with_suffix("").parts)
            for path in models_root.rglob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("_")
        }
        super().__init__()

    def __getitem__(self, key: str) -> type:
        if key in self:
            return super().__getitem__(key)
        if key not in self._map:
            raise KeyError(f"Model '{key}' not found. Available: {sorted(self._map)}")
        module = importlib.import_module(self._map[key])
        model_class = getattr(module, "Model", None) or getattr(module, key)
        self[key] = model_class
        return model_class

    def module_path(self, key: str) -> str:
        """Return the import path registered for a model name."""
        if key not in self._map:
            raise KeyError(f"Model '{key}' not found. Available: {sorted(self._map)}")
        return self._map[key]


class ForecastWindowDataset(Dataset):
    """Sliding-window dataset for long-term forecast models."""

    def __init__(
        self,
        split_payload: dict[str, np.ndarray],
        seq_len: int,
        label_len: int,
        pred_len: int,
        target_index: int,
    ) -> None:
        self.data = split_payload["data"].astype(np.float32)
        self.timestamps = split_payload["timestamps"]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_index = target_index

        available = len(self.data) - self.seq_len - self.pred_len + 1
        if available <= 0:
            raise ValueError(
                "Split is too short for the requested seq_len/pred_len window sizes"
            )
        self._length = available

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    def prediction_targets(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return timestamps and target values for the prediction horizon."""
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        target_timestamps = self.timestamps[s_end:r_end]
        target_values = self.data[s_end:r_end, self.target_index]
        return target_timestamps, target_values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="CastZoo experiment runner")
    parser.add_argument("--spec", required=True, help="Path to an experiment spec JSON")
    parser.add_argument("--out-dir", default="runs", help="Base directory for run artifacts")
    parser.add_argument(
        "--evaluator-hash-store",
        default=".forecast/evaluator.sha256",
        help="Path to the persisted evaluator hash file",
    )
    return parser.parse_args(argv)


def load_spec(spec_path: Path) -> tuple[dict[str, Any], str, str]:
    """Load the experiment spec and its canonical JSON representation."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    run_id = compute_spec_hash(spec)
    return spec, canonical, run_id


def get_target_index(spec: dict[str, Any]) -> int:
    """Resolve the target column index within the non-time feature matrix."""
    df = pd.read_csv(spec["dataset_path"], nrows=0)
    feature_cols = [column for column in df.columns if column != spec["time_col"]]
    if spec["target_col"] not in feature_cols:
        raise ValueError(f"Target column {spec['target_col']!r} not found in dataset")
    return feature_cols.index(spec["target_col"])


# Foundation models that use zero-shot inference and require task_name='zero_shot_forecast'
# in their forward() method.  All other models use the standard 'long_term_forecast' path.
ZERO_SHOT_MODELS: frozenset[str] = frozenset(
    ["Chronos", "Chronos2", "TimesFM", "Moirai", "Sundial", "TiRex", "TimeMoE"]
)


def build_model_config(spec: dict[str, Any], num_features: int) -> SimpleNamespace:
    """Build a minimal argparse-like config object for REF models."""
    hyperparams = dict(spec.get("hyperparams", {}))
    default_task_name = "zero_shot_forecast" if spec.get("model") in ZERO_SHOT_MODELS else "long_term_forecast"
    config_values: dict[str, Any] = {
        "task_name": default_task_name,
        "seq_len": spec["seq_len"],
        "label_len": spec.get("label_len", spec["seq_len"] // 2),
        "pred_len": spec["pred_len"],
        "enc_in": num_features,
        "dec_in": num_features,
        "c_out": hyperparams.get("c_out", num_features if num_features > 1 else 1),
        "d_model": hyperparams.get("d_model", 512),
        "n_heads": hyperparams.get("n_heads", 8),
        "e_layers": hyperparams.get("e_layers", 2),
        "d_layers": hyperparams.get("d_layers", 1),
        "d_ff": hyperparams.get("d_ff", 2048),
        "dropout": hyperparams.get("dropout", 0.1),
        "embed": hyperparams.get("embed", "timeF"),
        "freq": spec.get("freq", "h"),
        "top_k": hyperparams.get("top_k", 5),
        "num_kernels": hyperparams.get("num_kernels", 6),
        "moving_avg": hyperparams.get("moving_avg", 25),
        "individual": hyperparams.get("individual", False),
        "factor": hyperparams.get("factor", 3),
        "activation": hyperparams.get("activation", "gelu"),
        "num_class": hyperparams.get("num_class", 1),
        "output_attention": hyperparams.get("output_attention", False),
    }
    config_values.update(hyperparams)
    return SimpleNamespace(**config_values)


def build_decoder_input(seq_y: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Construct the decoder input expected by REF forecast models."""
    label_len = seq_y.shape[1] - pred_len
    zeros = torch.zeros(
        seq_y.shape[0],
        pred_len,
        seq_y.shape[2],
        dtype=seq_y.dtype,
        device=seq_y.device,
    )
    return torch.cat([seq_y[:, :label_len, :], zeros], dim=1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    pred_len: int,
    train_epochs: int,
    learning_rate: float,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> None:
    """Run a minimal supervised training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(train_epochs):
        epoch_losses: list[float] = []
        for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            seq_x_mark = seq_x_mark.to(device)
            seq_y_mark = seq_y_mark.to(device)

            decoder_input = build_decoder_input(seq_y, pred_len)
            target = seq_y[:, -pred_len:, :]

            optimizer.zero_grad(set_to_none=True)
            output = model(seq_x, seq_x_mark, decoder_input, seq_y_mark)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if on_epoch is not None:
            on_epoch(epoch, train_epochs, epoch_loss)
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")


def write_predictions(
    model: nn.Module,
    dataset: ForecastWindowDataset,
    run_dir: Path,
    device: torch.device,
    pred_len: int,
) -> Path:
    """Generate prediction rows for the evaluation split and write predictions.csv."""
    rows: list[dict[str, float | str]] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[index]
            seq_x = seq_x.unsqueeze(0).to(device)
            seq_y = seq_y.unsqueeze(0).to(device)
            seq_x_mark = seq_x_mark.unsqueeze(0).to(device)
            seq_y_mark = seq_y_mark.unsqueeze(0).to(device)

            decoder_input = build_decoder_input(seq_y, pred_len)
            output = model(seq_x, seq_x_mark, decoder_input, seq_y_mark)
            predictions = output[0, :, dataset.target_index].detach().cpu().numpy()

            timestamps, actual = dataset.prediction_targets(index)
            for timestamp, actual_value, predicted_value in zip(
                timestamps, actual, predictions, strict=True
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

    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    np.save(run_dir / "actual.npy", np.asarray(actual_values, dtype=np.float64), allow_pickle=False)
    np.save(run_dir / "pred.npy", np.asarray(predicted_values, dtype=np.float64), allow_pickle=False)
    return predictions_path


def write_classical_predictions(
    model: Any,
    train_payload: dict[str, np.ndarray],
    eval_payload: dict[str, np.ndarray],
    run_dir: Path,
    target_index: int,
    spec: dict[str, Any],
) -> Path:
    """Fit a statistical/ML wrapper and write evaluation predictions."""
    model.fit(train_payload=train_payload, target_index=target_index, spec=spec)
    predict_windows = getattr(model, "predict_windows", None)
    if callable(predict_windows):
        rows, actual, predictions = predict_windows(
            eval_payload=eval_payload,
            target_index=target_index,
            spec=spec,
            train_payload=train_payload,
        )
        predictions_path = run_dir / "predictions.csv"
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
        np.save(run_dir / "actual.npy", np.asarray(actual, dtype=np.float64), allow_pickle=False)
        np.save(run_dir / "pred.npy", np.asarray(predictions, dtype=np.float64), allow_pickle=False)
        return predictions_path

    predictions = model.predict(
        eval_payload=eval_payload,
        target_index=target_index,
        spec=spec,
        train_payload=train_payload,
    )
    predictions = np.asarray(predictions, dtype=np.float64)
    actual = eval_payload["data"][:, target_index].astype(np.float64)
    timestamps = eval_payload["timestamps"]

    if len(predictions) != len(actual):
        raise ValueError(
            f"Classical model returned {len(predictions)} predictions for {len(actual)} evaluation rows"
        )

    rows = [
        {
            "timestamp": str(timestamp),
            "actual": float(actual_value),
            "predicted": float(predicted_value),
        }
        for timestamp, actual_value, predicted_value in zip(
            timestamps, actual, predictions, strict=True
        )
    ]

    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    np.save(run_dir / "actual.npy", actual, allow_pickle=False)
    np.save(run_dir / "pred.npy", predictions, allow_pickle=False)
    return predictions_path


def get_model_family(module_path: str) -> str:
    """Resolve the top-level model family from the module import path."""
    if ".models.statistical_models." in module_path:
        return "statistical"
    if ".models.machine_learning_models." in module_path:
        return "machine_learning"
    if ".models.deep_learning_models." in module_path:
        return "deep_learning"
    if ".models.foundation_models." in module_path:
        return "foundation"
    return "other"


def run_neural_model(
    model_class: type[nn.Module],
    spec: dict[str, Any],
    dataset_splits: dict[str, dict[str, np.ndarray]],
    eval_split: str,
    target_index: int,
    device: torch.device,
    run_dir: Path,
    emit_event: Callable[..., None],
) -> Path:
    """Train and evaluate a neural forecast model."""
    num_features = dataset_splits["train"]["data"].shape[1]
    config = build_model_config(spec, num_features)
    model = model_class(config).to(device)

    train_dataset = ForecastWindowDataset(
        dataset_splits["train"],
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_index=target_index,
    )
    eval_dataset = ForecastWindowDataset(
        dataset_splits[eval_split],
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_index=target_index,
    )

    batch_size = int(spec.get("hyperparams", {}).get("batch_size", 32))
    learning_rate = float(spec.get("hyperparams", {}).get("learning_rate", 0.0001))
    train_epochs = int(spec.get("hyperparams", {}).get("train_epochs", 5))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        pred_len=config.pred_len,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        on_epoch=lambda epoch, total_epochs, loss: emit_event(
            "epoch",
            epoch=epoch + 1,
            total_epochs=total_epochs,
            loss=round(loss, 6),
        ),
    )

    emit_event("predict_start")
    return write_predictions(
        model=model,
        dataset=eval_dataset,
        run_dir=run_dir,
        device=device,
        pred_len=config.pred_len,
    )


def run_statistical_model(
    model_class: type,
    spec: dict[str, Any],
    dataset_splits: dict[str, dict[str, np.ndarray]],
    eval_split: str,
    target_index: int,
    run_dir: Path,
    emit_event: Callable[..., None],
) -> Path:
    """Fit and evaluate a statistical forecasting model."""
    config = build_model_config(spec, dataset_splits["train"]["data"].shape[1])
    model = model_class(config)
    emit_event("predict_start")
    return write_classical_predictions(
        model=model,
        train_payload=dataset_splits["train"],
        eval_payload=dataset_splits[eval_split],
        run_dir=run_dir,
        target_index=target_index,
        spec=spec,
    )


def run_machine_learning_model(
    model_class: type,
    spec: dict[str, Any],
    dataset_splits: dict[str, dict[str, np.ndarray]],
    eval_split: str,
    target_index: int,
    run_dir: Path,
    emit_event: Callable[..., None],
) -> Path:
    """Fit and evaluate a machine-learning forecasting model."""
    config = build_model_config(spec, dataset_splits["train"]["data"].shape[1])
    model = model_class(config)
    emit_event("predict_start")
    return write_classical_predictions(
        model=model,
        train_payload=dataset_splits["train"],
        eval_payload=dataset_splits[eval_split],
        run_dir=run_dir,
        target_index=target_index,
        spec=spec,
    )


def run_deep_learning_model(
    model_class: type[nn.Module],
    spec: dict[str, Any],
    dataset_splits: dict[str, dict[str, np.ndarray]],
    eval_split: str,
    target_index: int,
    device: torch.device,
    run_dir: Path,
    emit_event: Callable[..., None],
) -> Path:
    """Train and evaluate a deep-learning forecasting model."""
    return run_neural_model(
        model_class=model_class,
        spec=spec,
        dataset_splits=dataset_splits,
        eval_split=eval_split,
        target_index=target_index,
        device=device,
        run_dir=run_dir,
        emit_event=emit_event,
    )


def run_foundation_model(
    model_class: type[nn.Module],
    spec: dict[str, Any],
    dataset_splits: dict[str, dict[str, np.ndarray]],
    eval_split: str,
    target_index: int,
    device: torch.device,
    run_dir: Path,
    emit_event: Callable[..., None],
) -> Path:
    """Run a foundation-model wrapper with direct inference only."""
    num_features = dataset_splits["train"]["data"].shape[1]
    config = build_model_config(spec, num_features)
    model = model_class(config).to(device)

    eval_dataset = ForecastWindowDataset(
        dataset_splits[eval_split],
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_index=target_index,
    )
    emit_event("predict_start")
    return write_predictions(
        model=model,
        dataset=eval_dataset,
        run_dir=run_dir,
        device=device,
        pred_len=config.pred_len,
    )


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Execute the full experiment pipeline and return a structured result."""
    spec_path = Path(args.spec)
    spec, canonical_spec, run_id = load_spec(spec_path)
    run_dir = Path(args.out_dir) / run_id
    evaluator_hash_store = Path(args.evaluator_hash_store)

    cached_eval_path = run_dir / "eval.json"
    if cached_eval_path.exists():
        cached_eval = json.loads(cached_eval_path.read_text(encoding="utf-8"))
        return (
            {
                "status": "cached",
                "run_id": run_id,
                "out_dir": str(run_dir),
                "metrics": cached_eval.get("metrics"),
            },
            0,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(canonical_spec, encoding="utf-8")
    train_log_path = run_dir / "train.log"
    _orig_stderr = sys.stderr
    _run_start = _time.monotonic()

    def _emit_event(event_type: str, **kwargs: object) -> None:
        """Emit a JSON event line on the original stderr for TUI progress."""
        event = {
            "type": event_type,
            "elapsed_s": round(_time.monotonic() - _run_start, 2),
            **kwargs,
        }
        print(json.dumps(event), file=_orig_stderr, flush=True)

    with train_log_path.open("w", encoding="utf-8") as train_log:
        try:
            with contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(train_log):
                reject_if_test_in_forecast(spec)
                set_seed(spec.get("seed", 42))
                _emit_event("train_start")
                device = detect_device()
                print(f"device_info={json.dumps(get_device_info(), sort_keys=True)}")

                allowed_splits = ["train", "val"]
                if spec.get("phase") == "post-forecast":
                    allowed_splits = ["train", "val", "test"]

                dataset_splits = load_dataset(spec, allowed_splits=allowed_splits)
                eval_split = spec.get("eval_split", "val")
                if eval_split not in dataset_splits:
                    raise ValueError(f"Requested eval split {eval_split!r} is not available")

                target_index = get_target_index(spec)
                model_name = "ARIMA" if spec["model"] == "SARIMA" else spec["model"]
                model_registry = LazyModelDict(str(Path(__file__).resolve().parent / "models"))
                model_class = model_registry[model_name]
                model_family = get_model_family(model_registry.module_path(model_name))

                if model_family == "statistical":
                    predictions_path = run_statistical_model(
                        model_class=model_class,
                        spec=spec,
                        dataset_splits=dataset_splits,
                        eval_split=eval_split,
                        target_index=target_index,
                        run_dir=run_dir,
                        emit_event=_emit_event,
                    )
                elif model_family == "machine_learning":
                    predictions_path = run_machine_learning_model(
                        model_class=model_class,
                        spec=spec,
                        dataset_splits=dataset_splits,
                        eval_split=eval_split,
                        target_index=target_index,
                        run_dir=run_dir,
                        emit_event=_emit_event,
                    )
                elif model_family == "deep_learning":
                    if not issubclass(model_class, nn.Module):
                        raise TypeError(f"Deep-learning model '{model_name}' must inherit torch.nn.Module")
                    predictions_path = run_deep_learning_model(
                        model_class=model_class,
                        spec=spec,
                        dataset_splits=dataset_splits,
                        eval_split=eval_split,
                        target_index=target_index,
                        device=device,
                        run_dir=run_dir,
                        emit_event=_emit_event,
                    )
                elif model_family == "foundation":
                    if not issubclass(model_class, nn.Module):
                        raise TypeError(f"Foundation model '{model_name}' must inherit torch.nn.Module")
                    predictions_path = run_foundation_model(
                        model_class=model_class,
                        spec=spec,
                        dataset_splits=dataset_splits,
                        eval_split=eval_split,
                        target_index=target_index,
                        device=device,
                        run_dir=run_dir,
                        emit_event=_emit_event,
                    )
                else:
                    raise ValueError(
                        f"Model '{model_name}' belongs to unsupported family '{model_family}'"
                    )

                train_target = dataset_splits["train"]["data"][:, target_index].astype(np.float64)
                train_scale = float(np.mean(np.abs(np.diff(train_target)))) if len(train_target) > 1 else 0.0
                (run_dir / "train_scale.json").write_text(
                    json.dumps({"train_scale": train_scale}, indent=2),
                    encoding="utf-8",
                )

                eval_result = evaluate(
                    run_dir=run_dir,
                    spec=spec,
                    evaluator_hash_store=evaluator_hash_store,
                    predictions_path=predictions_path,
                    train_values=train_target,
                )
                _emit_event("done")

            return (
                {
                    "status": "success",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "metrics": eval_result["metrics"],
                },
                0,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traceback.print_exc(file=train_log)
            return (
                {
                    "status": "error",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                1,
            )
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc(file=train_log)
            return (
                {
                    "status": "error",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                1,
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    result, exit_code = run_experiment(parse_args(argv))
    print(json.dumps(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
