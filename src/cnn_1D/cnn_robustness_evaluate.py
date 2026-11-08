"""Evaluate the 12 original-width CNNs on fixed validation perturbations."""

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
OUTPUT = ROOT / "robustness/augmentation_robustness_v1.csv"

ORIGINAL = (16, 32, 64, 96)
SEEDS = (42, 1337, 2026)

MODES = {
    "none": "None",
    "shift_p0.5": "Shift",
    "noise_p0.5_20dB": "Noise",
    "shift_p0.5_noise_p0.5_20dB": "Both",
}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Results already exist: {OUTPUT}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verify that the saved benchmark matches the current
    # validation split, without accessing the test split.
    X_val, y_val = load_split("val")

    X_val = np.ascontiguousarray(
        X_val, dtype=np.float64
    )
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

    for name, X in conditions.items():
        assert X.shape == (27, 5000), name
        assert np.isfinite(X).all(), name

    label_map = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }

    targets = torch.tensor(
        [label_map[str(y)] for y in y_val],
        dtype=torch.long,
        device=device
    )

    checkpoints = {}

    paths = sorted(ROOT.glob("cnn_*_seed*/checkpoint.pth"))
    paths.extend([
        ROOT / "cnn_rough_seed42.pth",
        ROOT / "cnn_rough_aug_seed42.pth",
    ])

    # Identify experiments by checkpoint metadata,
    # rather than relying on directory naming alone.
    for path in paths:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )

        channels = tuple(ckpt.get("channels", ORIGINAL))
        dropout = float(ckpt.get("dropout", 0.20))
        lr = float(ckpt["learning_rate"])
        mode = ckpt.get("augmentation")
        seed = int(ckpt["seed"])

        if channels != ORIGINAL:
            continue
        if abs(dropout - 0.20) > 1e-9:
            continue
        if abs(lr - 0.001) > 1e-9:
            continue
        if mode not in MODES or seed not in SEEDS:
            continue

        assert int(ckpt["epoch_budget"]) == 60
        assert int(ckpt["batch_size"]) == 32
        assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)

        key = (mode, seed)

        if key in checkpoints:
            raise ValueError(f"Duplicate checkpoint: {key}")

        checkpoints[key] = (path, ckpt)

    expected = {
        (mode, seed)
        for mode in MODES
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

    for mode, label in MODES.items():
        for seed in SEEDS:
            path, ckpt = checkpoints[(mode, seed)]

            model = BearingCNN().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            mean = float(ckpt["train_mean"])
            std = float(ckpt["train_std"])

            assert std > 0

            for condition, X in sorted(conditions.items()):
                X_normalized = np.ascontiguousarray(
                    ((X - mean) / std)[:, None, :],
                    dtype=np.float32
                )

                inputs = torch.from_numpy(
                    X_normalized
                ).to(device)

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
                    "mode": label,
                    "seed": seed,
                    "checkpoint": str(path),
                    "condition": condition,
                    "samples": len(y_val),
                    "correct": correct,
                    "accuracy": correct / len(y_val),
                    "cross_entropy": loss,
                    "validation_sha256": fingerprint,
                })

            print(f"PASSED: {label}, seed {seed}")

    assert len(results) == 12 * 20

    # Write only after all models have been evaluated.
    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nAUGMENTATION ROBUSTNESS RESULTS")
    print("Mode | Clean accuracy | Mean 0 dB accuracy")

    for label in MODES.values():
        clean = [
            r["accuracy"] for r in results
            if r["mode"] == label
            and r["condition"] == "X_clean"
        ]

        severe_noise = [
            r["accuracy"] for r in results
            if r["mode"] == label
            and r["condition"].startswith("X_noise_0db_")
        ]

        print(
            f"{label} | {np.mean(clean):.4f} | "
            f"{np.mean(severe_noise):.4f}"
        )

    print("\nResults saved:", OUTPUT)
    print("Rows:", len(results))
    print("ALL ROBUSTNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
