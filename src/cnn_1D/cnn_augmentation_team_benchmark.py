"""Shared Colab CPU benchmark for original CNN augmentation variants."""

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

ROOT = Path("cnn_outputs")

SPECS = [
    ("original_reference_seed42", "none", "final_models"),
    ("original_shift_seed42", "shift_p0.5", "augmentation_models"),
    ("original_noise_seed42", "noise_p0.5_20dB", "augmentation_models"),
    ("original_both_seed42", "shift_p0.5_noise_p0.5_20dB",
     "augmentation_models"),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)

    manifests = {
        folder: {
            item["name"]: item
            for item in json.loads(
                (ROOT / folder / "manifest.json")
                .read_text(encoding="utf-8")
            )["models"]
        }
        for folder in ("final_models", "augmentation_models")
    }

    X_val, _ = load_split("val")
    assert X_val.shape == (27, 5000)

    results = []

    print("Device: CPU")
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("Threads:", torch.get_num_threads())
    print("Timing: batch=1, warmup=10, repetitions=100")

    for name, augmentation, folder in SPECS:
        path = ROOT / folder / f"{name}.pth"
        assert path.is_file(), path

        item = manifests[folder][name]
        assert sha256(path) == item["artifact_sha256"]
        assert int(item["seed"]) == 42

        package = torch.load(
            path, map_location="cpu", weights_only=True
        )

        assert int(package["seed"]) == 42
        assert tuple(package["channels"]) == (16, 32, 64, 96)
        assert tuple(package["class_names"]) == tuple(CLASS_NAMES)
        assert package.get("augmentation", "none") == augmentation

        model = BearingCNN(
            channels=tuple(package["channels"]),
            dropout=float(package["dropout"]),
        )
        model.load_state_dict(package["model_state_dict"])
        model.eval()

        parameters = sum(p.numel() for p in model.parameters())
        assert parameters == 50819

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
        assert abs(
            size_mb - item["size_bytes"] / 1_000_000
        ) < 1e-10

        latency = measure_inference_latency_ms(
            predict_one, X_val
        )

        results.append({
            "name": name,
            "augmentation": augmentation,
            "seed": 42,
            "parameters": parameters,
            "model_size_mb": size_mb,
            "mean_latency_ms": latency["mean_ms"],
            "std_latency_ms": latency["std_ms"],
            "artifact_sha256": sha256(path),
        })

        print(
            f"PASSED: {name} | "
            f"size={size_mb:.6f} MB | "
            f"latency={latency['mean_ms']:.4f} ms"
        )

    assert len(results) == 4

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

    print("\nAUGMENTATION BENCHMARK PASSED")
    print("Models measured: 4")
    print("Test set accessed: NO")


if __name__ == "__main__":
    main()
