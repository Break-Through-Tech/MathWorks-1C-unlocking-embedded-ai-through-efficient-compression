"""
src/cnn_1D/cnn_quantize.py

8-bit post-training static quantization (PTQ) for the BearingCNN baseline.

Pipeline
--------
1. Load the FP32 baseline checkpoint (default: cnn_outputs/cnn_rough_seed42.pth,
   the same checkpoint the pruning experiments used).
2. Wrap the network with QuantStub / DeQuantStub and fuse each
   Conv1d -> BatchNorm1d -> ReLU block into a single op. Fusing folds the
   BatchNorm into the conv weights, so it is not quantized separately.
3. Insert observers, then CALIBRATE: run training windows through the model
   so the observers can record activation ranges. Calibration uses the
   TRAINING split only. Validation is used to compare calibration settings,
   and test is used only for the final reported numbers.
4. Convert to int8: weights are stored as int8 and conv/linear math runs in
   int8. Input is quantized on entry and logits are dequantized on exit.
5. Evaluate FP32 and INT8 side by side in ONE process (same machine, same
   thread count) on accuracy, on-disk size and batch-1 latency, all using
   the shared functions in src/metrics.py.

Backend: 'fbgemm' on x86 (e.g. Google Colab CPU) and 'qnnpack' on ARM (e.g.
Apple Silicon). The script picks whichever this PyTorch build supports.

Usage (from the repository root):

    python -m src.cnn_1D.cnn_quantize                      # run + print results
    python -m src.cnn_1D.cnn_quantize --observer minmax    # other calibration
    python -m src.cnn_1D.cnn_quantize --save               # also append rows to
                                                           # results/experiments.csv
"""

import argparse
import copy
import json
import platform
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.ao import quantization as tq

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES
from src.metrics import accuracy, measure_inference_latency_ms, measure_model_size_mb

DEFAULT_CHECKPOINT = Path("cnn_outputs/cnn_rough_seed42.pth")
OUTPUT_DIR = Path("cnn_outputs/quantized")
CSV_PATH = Path("results/experiments.csv")
SEED = 42
AUTHOR = "Venya Goyal"

# torch.ao eager-mode quantization still works in torch 2.x but prints a
# migration notice (to torchao) on every call; hide it to keep output readable.
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class QuantizableBearingCNN(nn.Module):
    """BearingCNN with quant/dequant boundaries for eager-mode quantization."""

    def __init__(self, fp32_model: BearingCNN):
        super().__init__()
        self.quant = tq.QuantStub()
        self.features = copy.deepcopy(fp32_model.features)
        self.classifier = copy.deepcopy(fp32_model.classifier)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        return self.dequant(x)

    def fuse(self):
        # features = [Conv, BN, ReLU, Pool] x 4  -> fuse indices (0,1,2), (4,5,6), ...
        groups = [[str(i), str(i + 1), str(i + 2)] for i in range(0, 16, 4)]
        tq.fuse_modules(self.features, groups, inplace=True)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_backend() -> str:
    engines = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in engines:
            return engine
    raise RuntimeError(f"No int8 backend available (supported: {engines})")


def make_qconfig(backend: str, observer: str):
    """
    observer:
        'histogram' : PyTorch's default qconfig for the backend. Activations use
                      HistogramObserver, which picks the clipping range that
                      minimizes quantization error and is less sensitive to outliers.
        'minmax'    : activation range = plain min/max seen during calibration.
    Weights use the backend's default weight observer (int8, symmetric) either way.
    """
    default = tq.get_default_qconfig(backend)
    if observer == "histogram":
        return default
    if observer == "minmax":
        # qnnpack (ARM) can use the full 8-bit range. fbgemm/x86 use
        # reduce_range=True (7-bit activations) to avoid overflow on CPUs
        # without VNNI, which matches the backend's own default.
        reduce_range = backend != "qnnpack"
        return tq.QConfig(
            activation=tq.MinMaxObserver.with_args(reduce_range=reduce_range),
            weight=default.weight,
        )
    raise ValueError(f"Unknown observer: {observer!r}")


def load_fp32(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert tuple(checkpoint["class_names"]) == CLASS_NAMES
    model = BearingCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def prepare_split(split, mean, std):
    """Raw .mat split -> standardized float32 array (N, 1, 5000) + int labels."""
    X_raw, y_raw = load_split(split)
    X = np.ascontiguousarray(((X_raw - mean) / std)[:, None, :], dtype=np.float32)
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    y = np.array([label_map[str(lbl)] for lbl in y_raw], dtype=np.int64)
    return X, y


@torch.inference_mode()
def predict_logits(model, X, batch_size=64):
    out = [model(torch.from_numpy(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def quantize(fp32_model, X_calib, backend, observer, calib_batch=32):
    """Fuse -> prepare (observers) -> calibrate -> convert to int8."""
    torch.backends.quantized.engine = backend

    qmodel = QuantizableBearingCNN(fp32_model).eval().fuse()
    qmodel.qconfig = make_qconfig(backend, observer)
    tq.prepare(qmodel, inplace=True)

    # Calibration: forward passes only, no labels and no gradient updates.
    with torch.inference_mode():
        for i in range(0, len(X_calib), calib_batch):
            qmodel(torch.from_numpy(X_calib[i:i + calib_batch]))

    tq.convert(qmodel, inplace=True)
    return qmodel


def fused_fp32_reference(fp32_model):
    """FP32 model with the same Conv+BN+ReLU fusion and no quantization.
    This is a sanity check that fusion alone does not change predictions."""
    return QuantizableBearingCNN(fp32_model).eval().fuse()


def save_artifacts(fp32_model, qmodel, out_dir: Path, example):
    """
    Save the files that would actually ship. Two formats are saved for each model:
      *_state_dict.pth : weights only (same format as the team's baseline row)
      *_scripted.pt    : self-contained TorchScript, loadable without this repo
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fp32_state_dict": out_dir / "fp32_state_dict.pth",
        "int8_state_dict": out_dir / "int8_state_dict.pth",
        "fp32_scripted": out_dir / "fp32_scripted.pt",
        "int8_scripted": out_dir / "int8_scripted.pt",
    }
    torch.save(fp32_model.state_dict(), paths["fp32_state_dict"])
    torch.save(qmodel.state_dict(), paths["int8_state_dict"])
    with torch.inference_mode():
        torch.jit.save(torch.jit.trace(fp32_model, example), paths["fp32_scripted"])
        torch.jit.save(torch.jit.trace(qmodel, example), paths["int8_scripted"])
    return paths


def make_predict_fn(model):
    def predict_fn(x: np.ndarray):
        with torch.inference_mode():
            return model(torch.from_numpy(x)).argmax(dim=1)
    return predict_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--observer", choices=["histogram", "minmax"], default="histogram")
    parser.add_argument("--calib-size", type=int, default=None,
                        help="Number of training windows to calibrate on (default: all 393).")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch threads for latency timing (pruning rows used 1).")
    parser.add_argument("--save", action="store_true",
                        help="Append FP32 + INT8 rows to results/experiments.csv.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    backend = pick_backend()

    fp32_model, ckpt = load_fp32(args.checkpoint)
    mean, std = ckpt["train_mean"], ckpt["train_std"]

    X_train, _ = prepare_split("train", mean, std)
    X_val, y_val = prepare_split("val", mean, std)
    X_test, y_test = prepare_split("test", mean, std)

    # Calibration set: training windows only, a seeded random subset if requested.
    if args.calib_size is not None and args.calib_size < len(X_train):
        idx = np.random.default_rng(SEED).choice(len(X_train), args.calib_size, replace=False)
        X_calib = X_train[np.sort(idx)]
    else:
        X_calib = X_train
    print(f"Backend: {backend} | observer: {args.observer} | calibration windows: {len(X_calib)}")

    qmodel = quantize(fp32_model, X_calib, backend, args.observer)

    # --- Fusion sanity check -------------------------------------------------
    fused = fused_fp32_reference(fp32_model)
    fusion_diff = (predict_logits(fp32_model, X_val) - predict_logits(fused, X_val)).abs().max().item()
    print(f"Fusion check (FP32 vs fused FP32, max |logit diff| on val): {fusion_diff:.2e}")

    # --- Accuracy ------------------------------------------------------------
    results = {"backend": backend, "observer": args.observer, "calib_windows": len(X_calib),
               "checkpoint": str(args.checkpoint), "threads": args.threads,
               "machine": f"{platform.system()} {platform.machine()}", "torch": torch.__version__}

    for name, model in (("fp32", fp32_model), ("int8", qmodel)):
        for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            preds = predict_logits(model, X).argmax(1).numpy()
            results[f"{name}_{split}_acc"] = accuracy(y, preds)

    fp32_test_logits = predict_logits(fp32_model, X_test)
    int8_test_logits = predict_logits(qmodel, X_test)
    results["test_prediction_agreement"] = float(
        (fp32_test_logits.argmax(1) == int8_test_logits.argmax(1)).float().mean())
    results["test_max_abs_logit_diff"] = float((fp32_test_logits - int8_test_logits).abs().max())

    # --- Size ----------------------------------------------------------------
    run_dir = OUTPUT_DIR / f"static_int8_{args.observer}_{backend}"
    paths = save_artifacts(fp32_model, qmodel, run_dir, torch.from_numpy(X_val[:1]))
    for key, p in paths.items():
        results[f"{key}_mb"] = measure_model_size_mb(path=p)

    # Confirm the shipped int8 TorchScript file reloads and gives the same predictions.
    reloaded = torch.jit.load(str(paths["int8_scripted"]))
    reloaded_preds = predict_logits(reloaded, X_test).argmax(1)
    assert torch.equal(reloaded_preds, int8_test_logits.argmax(1)), "Reloaded int8 model disagrees!"

    # --- Latency (batch 1, metrics defaults, both timed in this process) ------
    for name, model in (("fp32", fp32_model), ("int8", qmodel)):
        lat = measure_inference_latency_ms(make_predict_fn(model), X_test)
        for k in ("mean_ms", "std_ms", "min_ms", "max_ms"):
            results[f"{name}_latency_{k}"] = lat[k]

    # --- Report ---------------------------------------------------------------
    r = results
    print("\n" + "=" * 66)
    print(f"{'Metric':<34}{'FP32':>14}{'INT8':>14}")
    print("-" * 66)
    print(f"{'Val accuracy (27)':<34}{r['fp32_val_acc']:>14.4f}{r['int8_val_acc']:>14.4f}")
    print(f"{'Test accuracy (102)':<34}{r['fp32_test_acc']:>14.4f}{r['int8_test_acc']:>14.4f}")
    print(f"{'Size: state_dict .pth (MB)':<34}{r['fp32_state_dict_mb']:>14.6f}{r['int8_state_dict_mb']:>14.6f}")
    print(f"{'Size: TorchScript .pt (MB)':<34}{r['fp32_scripted_mb']:>14.6f}{r['int8_scripted_mb']:>14.6f}")
    print(f"{'Latency mean (ms, batch 1)':<34}{r['fp32_latency_mean_ms']:>14.4f}{r['int8_latency_mean_ms']:>14.4f}")
    print(f"{'Latency std (ms)':<34}{r['fp32_latency_std_ms']:>14.4f}{r['int8_latency_std_ms']:>14.4f}")
    print("-" * 66)
    print(f"Size reduction (state_dict): {r['fp32_state_dict_mb'] / r['int8_state_dict_mb']:.2f}x smaller")
    print(f"Speed-up (mean latency):     {r['fp32_latency_mean_ms'] / r['int8_latency_mean_ms']:.2f}x")
    print(f"Test prediction agreement FP32 vs INT8: {r['test_prediction_agreement']:.4f}")
    print(f"Max |logit diff| on test:   {r['test_max_abs_logit_diff']:.4f}")
    print("=" * 66)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved artifacts + results.json to {run_dir}/")

    if args.save:
        append_rows(results, run_dir)


def append_rows(r, run_dir):
    df = pd.read_csv(CSV_PATH)
    next_id = int(df["experiment_id"].max()) + 1
    today = date.today().strftime("%m/%d/%Y")
    env = f"{r['machine']} CPU, {r['threads']} thread, torch {r['torch']}, backend {r['backend']}"
    common = dict(date=today, author=AUTHOR, model="1D CNN (BearingCNN)",
                  input_repr="raw waveform (1x5000)", standardization="Global z-score",
                  augmentation="None")
    rows = [
        dict(common, compression="None",
             accuracy_test=r["fp32_test_acc"], model_size_mb=r["fp32_state_dict_mb"],
             latency_ms_per_sample=r["fp32_latency_mean_ms"],
             notes=(f"Dense FP32 baseline ({Path(r['checkpoint']).name}) re-timed in the same session as its "
                    f"INT8 version. Size = fp32 state_dict .pth (TorchScript: {r['fp32_scripted_mb']:.6f} MB). "
                    f"val_acc={r['fp32_val_acc']:.4f}; {env}")),
        dict(common, compression="Quantization",
             accuracy_test=r["int8_test_acc"], model_size_mb=r["int8_state_dict_mb"],
             latency_ms_per_sample=r["int8_latency_mean_ms"],
             notes=(f"Post-training static INT8 quantization of {Path(r['checkpoint']).name}: Conv+BN+ReLU fused, "
                    f"per-tensor activations ({r['observer']} observer), calibrated on {r['calib_windows']} "
                    f"train windows, no retraining. Size = int8 state_dict .pth (TorchScript: "
                    f"{r['int8_scripted_mb']:.6f} MB). val_acc={r['int8_val_acc']:.4f}; "
                    f"FP32/INT8 test agreement={r['test_prediction_agreement']:.4f}; {env}. "
                    f"Files in {run_dir}")),
    ]
    for i, row in enumerate(rows):
        row["experiment_id"] = next_id + i
    df = pd.concat([df, pd.DataFrame(rows)[df.columns]], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Appended experiments {next_id}-{next_id + 1} to {CSV_PATH}")


if __name__ == "__main__":
    main()
