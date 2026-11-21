"""
src/cnn_1D/combine_projection_quantize.py

Projection + Quantization: take the 10%-rank projected model trained with
augmentation (cnn_outputs/projected/original_both_seed42/rank_10pct) and
quantize it to INT8 (static post-training quantization, calibrated on the
training split only).

conv3 and conv4 are pairs of thin convs after projection, so the
Conv+BN+ReLU fuse groups differ from Venya's baseline ones: only the second
conv of each pair fuses with its BatchNorm and ReLU (see
results/projection_writeup.md, Handoffs).

The shipped file is a small package (INT8 state dict + ranks + normalization),
loaded with load_projected_int8(path).

Usage (from the repository root):

    python -m src.cnn_1D.combine_projection_quantize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.ao import quantization as tq

from src.cnn_1D.cnn_model import CLASS_NAMES, BearingCNN
from src.cnn_1D.cnn_project import build_projected_model, load_model
from src.cnn_1D.cnn_quantize import (
    QuantizableBearingCNN, SEED, make_predict_fn, make_qconfig, pick_backend,
    predict_logits, prepare_split,
)
from src.cnn_1D.cnn_sparse import _int8_entries
from src.metrics import accuracy, measure_inference_latency_ms, measure_model_size_mb

SOURCE = Path("cnn_outputs/projected/original_both_seed42/rank_10pct/model.pth")
OUTPUT_DIR = Path("cnn_outputs/combined/projected10_int8")
FILE_NAME = "model_int8.pth"

# Fuse groups when conv3 (features.8) and conv4 (features.12) are split in two.
# Unprojected models (e.g. the micro model) use the baseline groups.
PLAIN_FUSE_GROUPS = [[str(i), str(i + 1), str(i + 2)] for i in range(0, 16, 4)]
FUSE_GROUPS = [["0", "1", "2"], ["4", "5", "6"], ["8.1", "9", "10"], ["12.1", "13", "14"]]


def _skeleton(fp32_model, backend, observer="histogram", groups=FUSE_GROUPS):
    """Fused, observer-prepared quantizable copy of a (projected) model."""
    torch.backends.quantized.engine = backend
    qmodel = QuantizableBearingCNN(fp32_model).eval()
    tq.fuse_modules(qmodel.features, groups, inplace=True)
    qmodel.qconfig = make_qconfig(backend, observer)
    tq.prepare(qmodel, inplace=True)
    return qmodel


def quantize_projected(fp32_model, X_calib, backend, calib_batch=32, groups=FUSE_GROUPS):
    qmodel = _skeleton(fp32_model, backend, groups=groups)
    with torch.inference_mode():
        for i in range(0, len(X_calib), calib_batch):
            qmodel(torch.from_numpy(X_calib[i:i + calib_batch]))
    tq.convert(qmodel, inplace=True)
    return qmodel


def save_projected_int8(qmodel, checkpoint, path):
    """Plain tensors only (int8 values + scales), so torch.load(weights_only=True) works."""
    int8, other = {}, {}
    for name, tensor in _int8_entries(qmodel).items():
        if tensor.is_quantized:
            assert tensor.qscheme() == torch.per_tensor_affine and tensor.q_zero_point() == 0
            int8[name] = (tensor.int_repr(), tensor.q_scale())
        else:
            other[name] = tensor
    torch.save({
        "format_version": 1,
        "int8_weights": int8,
        "other_tensors": other,
        "projection_ranks": checkpoint.get("projection_ranks", {}),
        "channels": list(checkpoint.get("channels", (16, 32, 64, 96))),
        "dropout": checkpoint.get("dropout", 0.20),
        "train_mean": checkpoint["train_mean"],
        "train_std": checkpoint["train_std"],
        "class_names": list(checkpoint["class_names"]),
    }, path)


def load_projected_int8(path, backend="qnnpack"):
    """Rebuild the quantized projected model from a file written above."""
    package = torch.load(path, map_location="cpu", weights_only=True)
    if package["projection_ranks"]:
        fp32, groups = build_projected_model(package["projection_ranks"]).eval(), FUSE_GROUPS
    else:
        fp32 = BearingCNN(channels=tuple(package["channels"]), dropout=package["dropout"]).eval()
        groups = PLAIN_FUSE_GROUPS
    model = _skeleton(fp32, backend, groups=groups)
    model(torch.zeros(1, 1, 5000))          # initialise observers before convert
    tq.convert(model, inplace=True)
    state = dict(package["other_tensors"])
    for name, (ints, scale) in package["int8_weights"].items():
        state[name] = torch._make_per_tensor_quantized_tensor(ints, scale, 0)
    model.load_state_dict(state)
    model.eval()
    return model, package


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    backend = pick_backend()

    fp32_model, ckpt = load_model(args.source)
    mean, std = ckpt["train_mean"], ckpt["train_std"]
    assert tuple(ckpt["class_names"]) == CLASS_NAMES
    X_train, _ = prepare_split("train", mean, std)
    X_val, y_val = prepare_split("val", mean, std)
    X_test, y_test = prepare_split("test", mean, std)

    qmodel = quantize_projected(fp32_model, X_train, backend)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / FILE_NAME
    save_projected_int8(qmodel, ckpt, path)
    reloaded, _ = load_projected_int8(path, backend)
    for X in (X_val, X_test):
        assert torch.equal(predict_logits(qmodel, X), predict_logits(reloaded, X)), "reload mismatch"

    results = {"backend": backend, "source": str(args.source), "file": str(path),
               "size_mb": measure_model_size_mb(path=path),
               "fp32_projected_size_mb": measure_model_size_mb(path=args.source)}
    for name, model in (("fp32_projected", fp32_model), ("combined", reloaded)):
        for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            results[f"{name}_{split}_acc"] = accuracy(y, predict_logits(model, X).argmax(1).numpy())
        lat = measure_inference_latency_ms(make_predict_fn(model), X_test)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms"):
            results[f"{name}_latency_{key}"] = lat[key]
    results["test_prediction_agreement"] = float(
        (predict_logits(fp32_model, X_test).argmax(1) == predict_logits(reloaded, X_test).argmax(1)).float().mean())

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
