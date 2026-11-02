"""
src/cnn_1D/projection_sweep.py

Projection Part 1 sweep: project both rough checkpoints (clean and
augmented) at several detail levels, before and after brief
fine-tuning, and save the validation results to
results/projection_sweep.csv for results/projection_plot.py.

Everything is measured in ONE process on ONE machine so the rows are
comparable. The test split is not used here; it is reserved for the
final Part 2 runs that go into results/experiments.csv.

Usage (from the repository root):

    python -m src.cnn_1D.projection_sweep
    python results/projection_plot.py
"""

import csv
from pathlib import Path

import torch

from src.cnn_1D.cnn_project import (
    DEFAULT_LAYERS, FINETUNE_EPOCHS, FINETUNE_LR, SEED,
    build_eval_loaders, finetune, load_model, measure, project_model,
)
from src.cnn_1D.cnn_train import set_seed

CHECKPOINTS = (
    Path("cnn_outputs/cnn_rough_seed42.pth"),
    Path("cnn_outputs/cnn_rough_aug_seed42.pth"),
)
KEEP_LEVELS = (0.75, 0.5, 0.25, 0.1, 0.05, 0.02)
OUTPUT_CSV = Path("results/projection_sweep.csv")
HARDWARE = "Windows laptop CPU, 1 thread"

# Short names for the two projected layers in the CSV.
LAYER_NAMES = {"features.8": "conv3", "features.12": "conv4"}


def make_row(checkpoint_path, checkpoint, keep, finetuned, summary, results, best_epoch):
    row = {
        "model": "augmented" if checkpoint.get("augmentation", "none") != "none" else "clean",
        "checkpoint": checkpoint_path.name,
        "keep": keep,
        "finetuned": finetuned,
        "finetune_best_epoch": best_epoch,
    }

    for layer in summary:
        short = LAYER_NAMES[layer["layer"]]
        row[f"{short}_rank"] = layer["rank"]
        row[f"{short}_energy_kept"] = round(layer["energy_kept"], 4)

    row.update({
        "params": results["params"],
        "size_mb": round(results["size_mb"], 6),
        "zipped_size_mb": round(results["zipped_size_mb"], 6),
        "latency_ms": round(results["latency"]["mean_ms"], 4),
        "val_accuracy": round(results["val_accuracy"], 4),
        "val_loss": results["val_loss"],
        "hardware": HARDWARE,
    })
    return row


def main():
    device = torch.device("cpu")
    rows = []

    for path in CHECKPOINTS:
        model, checkpoint = load_model(path, device)
        loaders = build_eval_loaders(checkpoint)

        # keep=1.0 is the unprojected baseline, measured in this same session.
        _, _, full_summary = project_model(model, DEFAULT_LAYERS, keep=1.0)
        rows.append(make_row(path, checkpoint, 1.0, False, full_summary,
                             measure(model, loaders, device), 0))

        for keep in KEEP_LEVELS:
            projected, _, summary = project_model(model, DEFAULT_LAYERS, keep=keep)
            rows.append(make_row(path, checkpoint, keep, False, summary,
                                 measure(projected, loaders, device), 0))

            set_seed(SEED)
            best_epoch = finetune(projected, checkpoint, loaders["val"],
                                  FINETUNE_EPOCHS, FINETUNE_LR, device)
            rows.append(make_row(path, checkpoint, keep, True, summary,
                                 measure(projected, loaders, device), best_epoch))

            before, after = rows[-2], rows[-1]
            print(
                f"{rows[-1]['model']:<9} keep={keep:<5} params={after['params']:>6} | "
                f"val acc {before['val_accuracy']:.2%} -> {after['val_accuracy']:.2%} | "
                f"val loss {before['val_loss']:.4f} -> {after['val_loss']:.4f}"
            )

    with OUTPUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
