"""Benchmark dense CNN inference bundles and CPU latency."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data import load_split
from src.metrics import (
    accuracy,
    measure_model_size_mb,
    measure_inference_latency_ms,
)
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs")
OUTPUT = ROOT / "benchmarks/dense_models_v1.csv"

WIDTHS = {
    "Original": (16, 32, 64, 96),
    "Medium": (12, 24, 48, 72),
    "Small": (8, 16, 32, 48),
    "Tiny": (4, 8, 16, 24),
    "Micro": (2, 4, 8, 12),
}
SEEDS = (42, 1337, 2026)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Results already exist: {OUTPUT}")

    torch.set_num_threads(1)
    device = torch.device("cpu")

    X_val, y_val = load_split("val")
    X_val = np.ascontiguousarray(X_val, dtype=np.float64)

    assert X_val.shape == (27, 5000)

    label_map = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }
    targets = np.array(
        [label_map[str(y)] for y in y_val], dtype=np.int64
    )

    reverse_widths = {
        channels: name for name, channels in WIDTHS.items()
    }

    checkpoints = {}
    paths = sorted(ROOT.glob("cnn_none_seed*/checkpoint.pth"))
    paths.append(ROOT / "cnn_rough_seed42.pth")

    for path in paths:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )

        channels = tuple(
            ckpt.get("channels", WIDTHS["Original"])
        )
        dropout = float(ckpt.get("dropout", 0.20))
        lr = float(ckpt["learning_rate"])
        seed = int(ckpt["seed"])

        if channels not in reverse_widths or seed not in SEEDS:
            continue
        if ckpt.get("augmentation") != "none":
            continue
        if abs(dropout - 0.20) > 1e-9:
            continue
        if abs(lr - 0.001) > 1e-9:
            continue

        assert int(ckpt["epoch_budget"]) == 60
        assert int(ckpt["batch_size"]) == 32
        assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)

        key = (reverse_widths[channels], seed)

        if key in checkpoints:
            raise ValueError(f"Duplicate: {key}")

        checkpoints[key] = (path, ckpt)

    expected = {
        (name, seed) for name in WIDTHS for seed in SEEDS
    }

    assert set(checkpoints) == expected, (
        f"Missing: {expected - set(checkpoints)}"
    )

    print("Device: CPU")
    print("Torch threads:", torch.get_num_threads())
    print("Checkpoints verified:", len(checkpoints))
    print("Latency: 10 warmup, 100 timed calls per model")
    print("Input: raw 5000-sample window")
    print()

    results = []

    for name, channels in WIDTHS.items():
        for seed in SEEDS:
            path, ckpt = checkpoints[(name, seed)]

            model = BearingCNN(
                channels=channels, dropout=0.20
            ).to(device)

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            n_params = sum(
                p.numel() for p in model.parameters()
            )

            mean = float(ckpt["train_mean"])
            std = float(ckpt["train_std"])
            assert std > 0

            # Candidate inference artifact includes the state
            # dict and metadata needed to reconstruct the model.
            package = {
                "model_state_dict": model.state_dict(),
                "channels": channels,
                "dropout": 0.20,
                "train_mean": mean,
                "train_std": std,
                "class_names": CLASS_NAMES,
            }

            with tempfile.TemporaryDirectory() as directory:
                package_path = Path(directory) / "model.pth"
                torch.save(package, package_path)

                size_mb = measure_model_size_mb(
                    path=package_path
                )

                reloaded = torch.load(
                    package_path,
                    map_location="cpu",
                    weights_only=True
                )
                restored = BearingCNN(
                    channels=tuple(reloaded["channels"]),
                    dropout=float(reloaded["dropout"])
                )
                restored.load_state_dict(
                    reloaded["model_state_dict"]
                )
                restored.eval()

                normalized = np.ascontiguousarray(
                    ((X_val - mean) / std)[:, None, :],
                    dtype=np.float32
                )

                with torch.inference_mode():
                    predictions = restored(
                        torch.from_numpy(normalized)
                    ).argmax(dim=1).numpy()

                val_accuracy = accuracy(targets, predictions)

            assert abs(
                val_accuracy - float(ckpt["val_accuracy"])
            ) < 1e-9

            # Timing includes preprocessing from raw signals,
            # tensor conversion, and CPU model inference.
            def predict_one(raw_sample):
                normalized = np.ascontiguousarray(
                    ((raw_sample - mean) / std)[:, None, :],
                    dtype=np.float32
                )
                with torch.inference_mode():
                    return model(
                        torch.from_numpy(normalized)
                    ).argmax(dim=1).numpy()

            latency = measure_inference_latency_ms(
                predict_one, X_val
            )

            results.append({
                "architecture": name,
                "seed": seed,
                "channels": str(channels),
                "parameters": n_params,
                "validation_accuracy": val_accuracy,
                "inference_bundle_size_mb": size_mb,
                "mean_latency_ms": latency["mean_ms"],
                "std_latency_ms": latency["std_ms"],
                "checkpoint": str(path),
                "device": "CPU",
                "torch_threads": 1,
                "input": "raw 5000-sample window",
                "includes_preprocessing": True,
            })

            print(
                f"{name:8} seed={seed} "
                f"params={n_params:6} "
                f"size={size_mb:.6f} MB "
                f"latency={latency['mean_ms']:.4f} ms "
                f"val_acc={val_accuracy:.2%}"
            )

    assert len(results) == 15

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nBENCHMARK COMPLETED")
    print("Results:", OUTPUT)
    print("Rows:", len(results))
    print("ALL DENSE BENCHMARK CHECKS PASSED")


if __name__ == "__main__":
    main()
