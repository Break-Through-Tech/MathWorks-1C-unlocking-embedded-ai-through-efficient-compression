"""Package the selected 1D CNN models for inference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs")
OUTPUT = ROOT / "final_models"

SPECS = [
    {
        "name": "original_reference_seed42",
        "path": ROOT / "cnn_rough_seed42.pth",
        "seed": 42,
        "channels": (16, 32, 64, 96),
        "parameters": 50819,
        "selection": "Established team compression reference",
    },
    {
        "name": "micro_selected_seed1337",
        "path": ROOT / (
            "cnn_none_seed1337_adam_lr0p001_bs32_ep60_"
            "m32faeaa30a_oe1f37b6114/checkpoint.pth"
        ),
        "seed": 1337,
        "channels": (2, 4, 8, 12),
        "parameters": 923,
        "selection": "Minimum clean-validation loss among micro seeds",
    },
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)

    manifest_path = OUTPUT / "manifest.json"

    outputs = [
        OUTPUT / f"{spec['name']}.pth"
        for spec in SPECS
    ] + [manifest_path]

    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")

    X_val, y_val = load_split("val")

    assert X_val.shape == (27, 5000)

    label_map = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    targets = np.array([
        label_map[str(y)] for y in y_val
    ])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "models": [],
    }

    for spec in SPECS:
        source = spec["path"]

        if not source.is_file():
            raise FileNotFoundError(source)

        ckpt = torch.load(
            source, map_location="cpu", weights_only=True
        )

        channels = tuple(
            ckpt.get("channels", (16, 32, 64, 96))
        )

        assert channels == spec["channels"]
        assert int(ckpt["seed"]) == spec["seed"]
        assert ckpt["augmentation"] == "none"
        assert abs(float(ckpt.get("dropout", 0.2)) - 0.2) < 1e-9
        assert abs(float(ckpt["learning_rate"]) - 0.001) < 1e-9
        assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)

        model = BearingCNN(
            channels=channels, dropout=0.2
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        parameter_count = sum(
            p.numel() for p in model.parameters()
        )
        assert parameter_count == spec["parameters"]

        train_mean = float(ckpt["train_mean"])
        train_std = float(ckpt["train_std"])
        assert np.isfinite(train_mean)
        assert np.isfinite(train_std) and train_std > 0

        package = {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "dropout": 0.2,
            "train_mean": train_mean,
            "train_std": train_std,
            "class_names": tuple(CLASS_NAMES),
            "seed": spec["seed"],
            "validation_loss": float(ckpt["val_loss"]),
            "validation_accuracy": float(ckpt["val_accuracy"]),
        }

        destination = OUTPUT / f"{spec['name']}.pth"

        with destination.open("xb") as file:
            torch.save(package, file)

        # Reload the exact serialized inference package.
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

        X = np.ascontiguousarray(
            ((X_val - train_mean) / train_std)[:, None, :],
            dtype=np.float32
        )
        inputs = torch.from_numpy(X)

        with torch.inference_mode():
            original_logits = model(inputs)
            restored_logits = restored(inputs)

        assert torch.equal(
            original_logits, restored_logits
        ), "Reloaded package changed model outputs"

        accuracy = float(
            (restored_logits.argmax(dim=1).numpy() == targets)
            .mean()
        )

        assert abs(
            accuracy - float(ckpt["val_accuracy"])
        ) < 1e-9

        item = {
            "name": spec["name"],
            "artifact": str(destination),
            "source_checkpoint": str(source),
            "source_sha256": digest(source),
            "artifact_sha256": digest(destination),
            "seed": spec["seed"],
            "parameters": parameter_count,
            "size_bytes": destination.stat().st_size,
            "validation_accuracy": accuracy,
            "validation_loss": float(ckpt["val_loss"]),
            "selection": spec["selection"],
        }

        manifest["models"].append(item)

        print(
            f"PASSED: {spec['name']} | "
            f"{parameter_count} parameters | "
            f"{item['size_bytes'] / 1_000_000:.6f} MB | "
            f"val_acc={accuracy:.2%}"
        )

    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    assert len(manifest["models"]) == 2

    print("\nFINAL MODEL PACKAGING PASSED")
    print("Models packaged: 2")
    print("Reload equivalence: PASSED")
    print("Validation accuracy: VERIFIED")
    print("Test set accessed: NO")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
