"""One-stage final evaluation of the 18 frozen CNN checkpoints."""

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data import load_split
from src.metrics import accuracy
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES

ROOT = Path("cnn_outputs")
OUTPUT = ROOT / "final_eval/frozen_test_results_v1.json"
FREEZE = Path("docs/cnn_final_selection_freeze.md")

SEEDS = (42, 1337, 2026)
ORIGINAL = (16, 32, 64, 96)
MICRO = (2, 4, 8, 12)

GROUPS = [
    ("Original clean", ORIGINAL, 0.20, "none"),
    ("Original shift", ORIGINAL, 0.20, "shift_p0.5"),
    ("Original noise", ORIGINAL, 0.20, "noise_p0.5_20dB"),
    ("Original both", ORIGINAL, 0.20,
     "shift_p0.5_noise_p0.5_20dB"),
    ("Original no dropout", ORIGINAL, 0.00, "none"),
    ("Micro clean", MICRO, 0.20, "none"),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Final test results already exist: {OUTPUT}"
        )

    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    paths = sorted(
        ROOT.glob("cnn_*_seed*/checkpoint.pth")
    )
    paths.extend([
        ROOT / "cnn_rough_seed42.pth",
        ROOT / "cnn_rough_aug_seed42.pth",
    ])

    expected = {
        (name, seed)
        for name, _, _, _ in GROUPS
        for seed in SEEDS
    }

    checkpoints = {}

    # Complete metadata checks before loading test data.
    for path in paths:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )

        channels = tuple(
            ckpt.get("channels", ORIGINAL)
        )
        dropout = float(ckpt.get("dropout", 0.20))
        augmentation = ckpt["augmentation"]
        seed = int(ckpt["seed"])

        for name, required_channels, required_dropout, mode in GROUPS:
            if (
                channels == required_channels
                and abs(dropout - required_dropout) < 1e-9
                and augmentation == mode
                and seed in SEEDS
                and abs(float(ckpt["learning_rate"]) - 0.001) < 1e-9
            ):
                key = (name, seed)

                if key in checkpoints:
                    raise ValueError(
                        f"Duplicate frozen checkpoint: {key}"
                    )

                assert int(ckpt["epoch_budget"]) == 60
                assert int(ckpt["batch_size"]) == 32
                assert tuple(ckpt["class_names"]) == tuple(CLASS_NAMES)
                assert float(ckpt["train_std"]) > 0

                checkpoints[key] = (
                    path, ckpt, channels, dropout
                )
                break

    if set(checkpoints) != expected:
        raise ValueError(
            f"Missing: {expected - set(checkpoints)}"
        )

    print("FROZEN FINAL TEST EVALUATION")
    print("Checkpoints verified:", len(checkpoints))
    print("Device:", device)
    print("Test data accessed only after checkpoint verification")

    # Access the test split only for final evaluation.
    X_test, y_test = load_split("test")

    assert X_test.shape == (102, 5000)
    assert len(y_test) == 102

    label_map = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }

    targets = np.asarray(
        [label_map[str(y)] for y in y_test],
        dtype=np.int64
    )

    assert all(
        np.sum(targets == i) == 34
        for i in range(len(CLASS_NAMES))
    )

    results = []

    for name, _, _, _ in GROUPS:
        for seed in SEEDS:
            path, ckpt, channels, dropout = checkpoints[
                (name, seed)
            ]

            model = BearingCNN(
                channels=channels,
                dropout=dropout
            ).to(device)

            model.load_state_dict(
                ckpt["model_state_dict"]
            )
            model.eval()

            train_mean = float(ckpt["train_mean"])
            train_std = float(ckpt["train_std"])

            X = np.ascontiguousarray(
                ((X_test - train_mean) / train_std)[:, None, :],
                dtype=np.float32
            )

            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(X)
                )
                predictions = (
                    logits.argmax(dim=1).numpy()
                )

            test_accuracy = accuracy(
                targets, predictions
            )

            cm = confusion_matrix(
                targets,
                predictions,
                labels=[0, 1, 2]
            )

            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    targets,
                    predictions,
                    labels=[0, 1, 2],
                    zero_division=0
                )
            )

            assert int(cm.sum()) == 102
            assert abs(
                test_accuracy - np.trace(cm) / 102
            ) < 1e-12

            per_class = {
                class_name: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, class_name in enumerate(CLASS_NAMES)
            }

            result = {
                "group": name,
                "seed": seed,
                "checkpoint": str(path),
                "channels": list(channels),
                "dropout": dropout,
                "augmentation": ckpt["augmentation"],
                "best_validation_epoch": int(ckpt["epoch"]),
                "test_accuracy": float(test_accuracy),
                "macro_precision": float(np.mean(precision)),
                "macro_recall": float(np.mean(recall)),
                "macro_f1": float(np.mean(f1)),
                "confusion_matrix": cm.tolist(),
                "per_class": per_class,
            }

            results.append(result)

            print(
                f"{name:20} seed={seed} "
                f"accuracy={test_accuracy:.2%} "
                f"macro_F1={np.mean(f1):.4f}"
            )

    assert len(results) == 18

    summaries = []

    print("\nTHREE-SEED TEST SUMMARY")
    print("Group | Mean accuracy | SD accuracy | Mean macro-F1")

    for name, _, _, _ in GROUPS:
        selected = [
            r for r in results
            if r["group"] == name
        ]

        accuracies = [
            r["test_accuracy"] for r in selected
        ]
        f1_scores = [
            r["macro_f1"] for r in selected
        ]

        summary = {
            "group": name,
            "mean_test_accuracy": mean(accuracies),
            "sd_test_accuracy": stdev(accuracies),
            "mean_macro_f1": mean(f1_scores),
            "sd_macro_f1": stdev(f1_scores),
        }

        summaries.append(summary)

        print(
            f"{name} | "
            f"{summary['mean_test_accuracy']:.4%} | "
            f"{summary['sd_test_accuracy']:.4%} | "
            f"{summary['mean_macro_f1']:.4f}"
        )

    report = {
        "evaluation": "Final frozen CNN test evaluation",
        "freeze_document": str(FREEZE),
        "test_samples": 102,
        "class_names": list(CLASS_NAMES),
        "seeds": list(SEEDS),
        "runs": results,
        "summaries": summaries,
        "limitations": [
            "Previous team test evaluations mean this is not fully blind.",
            "Recording-level split independence is unverified.",
            "The three training seeds share the same test examples.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite existing final results.
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nFinal report:", OUTPUT)
    print("Evaluated checkpoints:", len(results))
    print("FINAL TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
