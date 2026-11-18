"""Package the three original-CNN augmentation variants for benchmarking."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs")
OUTPUT = ROOT / "augmentation_models"

SPECS = [
    ("Original shift", "original_shift_seed42", "shift_p0.5"),
    ("Original noise", "original_noise_seed42", "noise_p0.5_20dB"),
    ("Original both", "original_both_seed42",
     "shift_p0.5_noise_p0.5_20dB"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)

    report = json.loads(
        (ROOT / "final_eval/frozen_test_results_v1.json")
        .read_text(encoding="utf-8")
    )

    selected = {}
    for group, name, mode in SPECS:
        matches = [
            r for r in report["runs"]
            if r["group"] == group and r["seed"] == 42
        ]
        assert len(matches) == 1
        run = matches[0]

        assert run["augmentation"] == mode
        assert run["test_accuracy"] == 1.0

        source = Path(run["checkpoint"])
        assert source.is_file(), source

        selected[name] = (source, mode)

    destinations = [
        OUTPUT / f"{name}.pth"
        for _, name, _ in SPECS
    ]
    destinations.append(OUTPUT / "manifest.json")

    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")

    X_val, y_val = load_split("val")
    assert X_val.shape == (27, 5000)

    label_map = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }
    targets = np.array([label_map[str(y)] for y in y_val])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "models": []}

    for group, name, mode in SPECS:
        source, _ = selected[name]
        ckpt = torch.load(
            source, map_location="cpu", weights_only=True
        )

        channels = tuple(
            ckpt.get("channels", (16, 32, 64, 96))
        )
        dropout = float(ckpt.get("dropout", 0.20))

        assert channels == (16, 32, 64, 96)
        assert abs(dropout - 0.20) < 1e-9
        assert int(ckpt["seed"]) == 42
        assert ckpt["augmentation"] == mode
        assert abs(float(ckpt["learning_rate"]) - 0.001) < 1e-9
        assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)

        model = BearingCNN(channels=channels, dropout=dropout)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        parameters = sum(p.numel() for p in model.parameters())
        assert parameters == 50819

        mean = float(ckpt["train_mean"])
        std = float(ckpt["train_std"])
        assert np.isfinite(mean)
        assert np.isfinite(std) and std > 0

        package = {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "dropout": dropout,
            "train_mean": mean,
            "train_std": std,
            "class_names": tuple(CLASS_NAMES),
            "seed": 42,
            "validation_loss": float(ckpt["val_loss"]),
            "validation_accuracy": float(ckpt["val_accuracy"]),
            "augmentation": mode,
        }

        destination = OUTPUT / f"{name}.pth"
        with destination.open("xb") as file:
            torch.save(package, file)

        restored_package = torch.load(
            destination, map_location="cpu", weights_only=True
        )
        restored = BearingCNN(
            channels=tuple(restored_package["channels"]),
            dropout=float(restored_package["dropout"])
        )
        restored.load_state_dict(
            restored_package["model_state_dict"]
        )
        restored.eval()

        inputs = torch.from_numpy(np.ascontiguousarray(
            ((X_val - mean) / std)[:, None, :],
            dtype=np.float32
        ))

        with torch.inference_mode():
            reference_logits = model(inputs)
            restored_logits = restored(inputs)

        assert torch.equal(reference_logits, restored_logits)

        val_acc = float(
            (restored_logits.argmax(dim=1).numpy() == targets).mean()
        )
        assert abs(val_acc - float(ckpt["val_accuracy"])) < 1e-9

        manifest["models"].append({
            "name": name,
            "group": group,
            "augmentation": mode,
            "seed": 42,
            "parameters": parameters,
            "source_checkpoint": str(source),
            "source_sha256": sha256(source),
            "artifact": str(destination),
            "artifact_sha256": sha256(destination),
            "size_bytes": destination.stat().st_size,
            "validation_accuracy": val_acc,
        })

        print(
            f"PASSED: {name} | "
            f"size={destination.stat().st_size / 1e6:.6f} MB | "
            f"val_acc={val_acc:.2%}"
        )

    assert len(manifest["models"]) == 3

    with (OUTPUT / "manifest.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print("\nAUGMENTATION PACKAGING PASSED")
    print("Models packaged: 3")
    print("Reload equivalence: PASSED")
    print("Test set accessed: NO")


if __name__ == "__main__":
    main()
