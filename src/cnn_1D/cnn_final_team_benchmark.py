"""Shared CPU benchmarking of the two final 1D CNN inference packages."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from src.data import load_split
from src.metrics import (
    measure_model_size_mb,
    measure_inference_latency_ms,
)
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs/final_models")

MODELS = [
    ("original_reference_seed42", (16, 32, 64, 96), 42),
    ("micro_selected_seed1337", (2, 4, 8, 12), 1337),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)

    manifest = json.loads(
        (ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_models = {
        item["name"]: item for item in manifest["models"]
    }

    assert set(manifest_models) == {
        name for name, _, _ in MODELS
    }

    X_val, _ = load_split("val")
    assert X_val.shape == (27, 5000)

    print("Device: CPU")
    print("PyTorch:", torch.__version__)
    print("Platform:", platform.platform())
    print("Threads:", torch.get_num_threads())
    print("Timing: 10 warmup, 100 repetitions, batch size 1")

    results = []

    for name, channels, seed in MODELS:
        path = ROOT / f"{name}.pth"
        item = manifest_models[name]

        assert path.is_file()
        assert sha256(path) == item["artifact_sha256"]

        package = torch.load(
            path, map_location="cpu", weights_only=True
        )

        assert tuple(package["channels"]) == channels
        assert int(package["seed"]) == seed
        assert tuple(package["class_names"]) == tuple(CLASS_NAMES)

        model = BearingCNN(
            channels=channels,
            dropout=float(package["dropout"])
        )
        model.load_state_dict(package["model_state_dict"])
        model.eval()

        mean = float(package["train_mean"])
        std = float(package["train_std"])
        assert np.isfinite(mean)
        assert np.isfinite(std) and std > 0

        def predict_one(raw_sample):
            normalized = np.ascontiguousarray(
                ((raw_sample - mean) / std)[:, None, :],
                dtype=np.float32
            )
            with torch.inference_mode():
                return model(
                    torch.from_numpy(normalized)
                ).argmax(dim=1).numpy()

        size_mb = measure_model_size_mb(path=path)

        latency = measure_inference_latency_ms(
            predict_one, X_val
        )

        results.append({
            "name": name,
            "seed": seed,
            "parameters": item["parameters"],
            "model_size_mb": size_mb,
            "mean_latency_ms": latency["mean_ms"],
            "std_latency_ms": latency["std_ms"],
            "artifact_sha256": sha256(path),
        })

        print(
            f"PASSED: {name} | "
            f"size={size_mb:.6f} MB | "
            f"mean_latency={latency['mean_ms']:.4f} ms"
        )

    assert len(results) == 2

    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)

        args.output.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "device": "CPU",
            "threads": torch.get_num_threads(),
            "timing_input": "raw validation windows",
            "includes_preprocessing": True,
            "warmup": 10,
            "repetitions": 100,
            "results": results,
        }

        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("Report:", args.output)

    print("\nFINAL SHARED BENCHMARK CHECKS PASSED")
    print("Models measured: 2")
    print("Test set accessed: NO")


if __name__ == "__main__":
    main()
