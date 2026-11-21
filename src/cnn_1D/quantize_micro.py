"""
src/cnn_1D/quantize_micro.py

Static INT8 quantization of Gurbaj's micro model
(cnn_outputs/final_models/micro_selected_seed1337.pth, channels 2-4-8-12,
923 parameters). Same pipeline as the other combined models; the file is
read back with load_projected_int8() from combine_projection_quantize.py.

Usage (from the repository root):

    python -m src.cnn_1D.quantize_micro
"""

import json
from pathlib import Path

import torch

from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_quantize import (
    SEED, make_predict_fn, pick_backend, predict_logits, prepare_split,
)
from src.cnn_1D.combine_projection_quantize import (
    PLAIN_FUSE_GROUPS, load_projected_int8, quantize_projected, save_projected_int8,
)
from src.metrics import accuracy, measure_inference_latency_ms, measure_model_size_mb

SOURCE = Path("cnn_outputs/final_models/micro_selected_seed1337.pth")
OUTPUT_DIR = Path("cnn_outputs/combined/micro_int8")


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    backend = pick_backend()

    ckpt = torch.load(SOURCE, map_location="cpu", weights_only=True)
    fp32 = BearingCNN(channels=tuple(ckpt.get("channels", (16, 32, 64, 96))),
                      dropout=ckpt.get("dropout", 0.20))
    fp32.load_state_dict(ckpt["model_state_dict"])
    fp32.eval()

    mean, std = ckpt["train_mean"], ckpt["train_std"]
    X_train, _ = prepare_split("train", mean, std)
    X_val, y_val = prepare_split("val", mean, std)
    X_test, y_test = prepare_split("test", mean, std)

    qmodel = quantize_projected(fp32, X_train, backend, groups=PLAIN_FUSE_GROUPS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "model_int8.pth"
    save_projected_int8(qmodel, ckpt, path)
    reloaded, _ = load_projected_int8(path, backend)
    for X in (X_val, X_test):
        assert torch.equal(predict_logits(qmodel, X), predict_logits(reloaded, X)), "reload mismatch"

    results = {"backend": backend, "source": str(SOURCE), "file": str(path),
               "size_mb": measure_model_size_mb(path=path),
               "fp32_micro_size_mb": measure_model_size_mb(path=SOURCE)}
    for name, model in (("fp32_micro", fp32), ("combined", reloaded)):
        for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            results[f"{name}_{split}_acc"] = accuracy(y, predict_logits(model, X).argmax(1).numpy())
        lat = measure_inference_latency_ms(make_predict_fn(model), X_test)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms"):
            results[f"{name}_latency_{key}"] = lat[key]
    results["test_prediction_agreement"] = float(
        (predict_logits(fp32, X_test).argmax(1) == predict_logits(reloaded, X_test).argmax(1)).float().mean())

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
