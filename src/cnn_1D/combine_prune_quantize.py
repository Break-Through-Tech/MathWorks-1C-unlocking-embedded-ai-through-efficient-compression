"""
src/cnn_1D/combine_prune_quantize.py

Pruning + Quantization: take the best pruned model (97% unstructured, 100%
test accuracy) and quantize it to INT8 with Venya's QuantizableBearingCNN
pipeline (src/cnn_1D/cnn_quantize.py), then ship it as sparse INT8
(nonzero int8 weights + zero-pattern mask, cnn_sparse.save_sparse_int8)
so the pruning saving is kept.

Checks that the sparse file reloads to a model with identical logits, then
measures test accuracy, shipped size and batch-1 latency.

Usage (from the repository root):

    python -m src.cnn_1D.combine_prune_quantize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.cnn_1D.cnn_quantize import (
    SEED, make_predict_fn, pick_backend, predict_logits, prepare_split, quantize,
)
from src.cnn_1D.cnn_sparse import load_sparse, load_sparse_int8, save_sparse_int8
from src.metrics import accuracy, measure_inference_latency_ms, measure_model_size_mb

PRUNED = Path("cnn_outputs/pruned/unstructured_97pct/model_sparse.npz")
OUTPUT_DIR = Path("cnn_outputs/combined/pruned97_int8")
FILE_NAME = "model_sparse_int8.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruned", type=Path, default=PRUNED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    backend = pick_backend()

    fp32_model, meta = load_sparse(args.pruned)
    mean, std = meta["train_mean"], meta["train_std"]
    X_train, _ = prepare_split("train", mean, std)
    X_val, y_val = prepare_split("val", mean, std)
    X_test, y_test = prepare_split("test", mean, std)

    qmodel = quantize(fp32_model, X_train, backend, "histogram")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / FILE_NAME
    save_sparse_int8(qmodel, meta, path)
    reloaded, _ = load_sparse_int8(path, backend)

    # The shipped file must reproduce the in-memory quantized model exactly.
    for X in (X_val, X_test):
        assert torch.equal(predict_logits(qmodel, X), predict_logits(reloaded, X)), "reload mismatch"

    weights = [m.weight().int_repr() for m in qmodel.modules() if hasattr(m, "weight") and callable(m.weight)]
    zeros = sum(int((w == 0).sum()) for w in weights)
    total = sum(w.numel() for w in weights)

    results = {
        "backend": backend,
        "source": str(args.pruned),
        "file": str(path),
        "int8_weight_zero_fraction": zeros / total,
        "size_mb": measure_model_size_mb(path=path),
        "fp32_pruned_sparse_mb": measure_model_size_mb(path=args.pruned),
    }
    for name, model in (("fp32_pruned", fp32_model), ("combined", reloaded)):
        for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            preds = predict_logits(model, X).argmax(1).numpy()
            results[f"{name}_{split}_acc"] = accuracy(y, preds)
        lat = measure_inference_latency_ms(make_predict_fn(model), X_test)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms"):
            results[f"{name}_latency_{key}"] = lat[key]
    results["test_prediction_agreement"] = float(
        (predict_logits(fp32_model, X_test).argmax(1) == predict_logits(reloaded, X_test).argmax(1)).float().mean())

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
