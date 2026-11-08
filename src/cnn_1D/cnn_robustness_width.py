"""Compare five CNN widths on the fixed validation robustness dataset."""

import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs")
INPUT = ROOT / "robustness/validation_perturbations_v1.npz"
OUTPUT = ROOT / "robustness/width_robustness_v1.csv"

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_val, y_val = load_split("val")
    X_val = np.ascontiguousarray(X_val, dtype=np.float64)
    fingerprint = sha256(X_val.tobytes()).hexdigest()

    with np.load(INPUT, allow_pickle=False) as saved:
        assert saved["validation_sha256"].item() == fingerprint
        assert np.array_equal(saved["X_clean"], X_val)
        assert np.array_equal(saved["y"], y_val)
        conditions = {
            key: saved[key].copy()
            for key in saved.files
            if key.startswith("X_")
        }

    assert len(conditions) == 20
    assert all(
        X.shape == (27, 5000) and np.isfinite(X).all()
        for X in conditions.values()
    )

    label_map = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }
    targets = torch.tensor(
        [label_map[str(y)] for y in y_val],
        dtype=torch.long,
        device=device
    )

    checkpoints = {}
    reverse_widths = {
        channels: name for name, channels in WIDTHS.items()
    }

    paths = sorted(ROOT.glob("cnn_*_seed*/checkpoint.pth"))
    paths.append(ROOT / "cnn_rough_seed42.pth")

    for path in paths:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )

        channels = tuple(
            ckpt.get("channels", WIDTHS["Original"])
        )
        seed = int(ckpt["seed"])

        if channels not in reverse_widths:
            continue
        if seed not in SEEDS:
            continue
        if ckpt.get("augmentation") != "none":
            continue
        if abs(float(ckpt.get("dropout", 0.2)) - 0.2) > 1e-9:
            continue
        if abs(float(ckpt["learning_rate"]) - 0.001) > 1e-9:
            continue

        assert int(ckpt["epoch_budget"]) == 60
        assert int(ckpt["batch_size"]) == 32
        assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)

        name = reverse_widths[channels]
        key = (name, seed)

        if key in checkpoints:
            raise ValueError(f"Duplicate checkpoint: {key}")

        checkpoints[key] = (path, ckpt)

    expected = {
        (name, seed)
        for name in WIDTHS
        for seed in SEEDS
    }

    assert set(checkpoints) == expected, (
        f"Missing: {expected - set(checkpoints)}"
    )

    print("Device:", device)
    print("Validation SHA256:", fingerprint)
    print("Checkpoints verified:", len(checkpoints))
    print("Conditions:", len(conditions))

    results = []

    for name, channels in WIDTHS.items():
        for seed in SEEDS:
            path, ckpt = checkpoints[(name, seed)]

            model = BearingCNN(channels=channels).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            params = sum(p.numel() for p in model.parameters())
            mean = float(ckpt["train_mean"])
            std = float(ckpt["train_std"])

            assert std > 0

            for condition, X in sorted(conditions.items()):
                normalized = np.ascontiguousarray(
                    ((X - mean) / std)[:, None, :],
                    dtype=np.float32
                )

                inputs = torch.from_numpy(normalized).to(device)

                with torch.inference_mode():
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits, targets
                    ).item()
                    correct = int(
                        (logits.argmax(dim=1) == targets)
                        .sum().item()
                    )

                results.append({
                    "architecture": name,
                    "seed": seed,
                    "channels": str(channels),
                    "parameters": params,
                    "checkpoint": str(path),
                    "condition": condition,
                    "samples": len(y_val),
                    "correct": correct,
                    "accuracy": correct / len(y_val),
                    "cross_entropy": loss,
                    "validation_sha256": fingerprint,
                })

            print(f"PASSED: {name}, seed {seed}")

    assert len(results) == 300

    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nARCHITECTURE ROBUSTNESS SUMMARY")
    print("Width | Clean | 10 dB | 5 dB | 0 dB | Shifts")

    groups = {
        "Clean": ["X_clean"],
        "10 dB": [f"X_noise_10db_r{r}" for r in range(3)],
        "5 dB": [f"X_noise_5db_r{r}" for r in range(3)],
        "0 dB": [f"X_noise_0db_r{r}" for r in range(3)],
        "Shifts": [
            "X_shift_m500", "X_shift_m250",
            "X_shift_p250", "X_shift_p500"
        ],
    }

    for name in WIDTHS:
        values = []
        for condition_keys in groups.values():
            selected = [
                r["accuracy"]
                for r in results
                if r["architecture"] == name
                and r["condition"] in condition_keys
            ]
            assert len(selected) == 3 * len(condition_keys)
            values.append(np.mean(selected) * 100)

        print(
            name + " | " +
            " | ".join(f"{value:.2f}%" for value in values)
        )

    print("\nResults saved:", OUTPUT)
    print("Rows:", len(results))
    print("ALL WIDTH ROBUSTNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
