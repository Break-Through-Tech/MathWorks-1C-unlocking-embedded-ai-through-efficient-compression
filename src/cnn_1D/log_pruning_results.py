"""
src/cnn_1D/log_pruning_results.py

Log the pruning experiment grid into results/experiments.csv.

Re-measures the dense baseline and every pruned checkpoint under
cnn_outputs/pruned/ in ONE process on ONE machine, so the 9 rows are
directly comparable (latency in particular varies across environments).
Appends one row per model to results/experiments.csv in the team's
column format.

Usage (from the repository root):

    python -m src.cnn_1D.log_pruning_results            # preview only
    python -m src.cnn_1D.log_pruning_results --save     # append to CSV
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_prune import build_eval_loaders, measure

BASELINE_CHECKPOINT = Path("cnn_outputs/cnn_rough_seed42.pth")
PRUNED_DIR = Path("cnn_outputs/pruned")
CSV_PATH = Path("results/experiments.csv")

AUTHOR = "Aigerim Zhadikbay"
DATE = "10/27/2026"
HARDWARE = "MacBook Air CPU, 1 thread"


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BearingCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def collect_rows():
    device = torch.device("cpu")

    _, baseline_ckpt = load_model(BASELINE_CHECKPOINT)
    loaders = build_eval_loaders(baseline_ckpt, include_test=True)

    # Dense baseline first, then pruned runs in a fixed, readable order.
    runs = [("none", None, BASELINE_CHECKPOINT)]
    for method in ("unstructured", "structured"):
        for level in (25, 50, 75, 90):
            path = PRUNED_DIR / f"{method}_{level}pct" / "checkpoint.pth"
            if path.exists():
                runs.append((method, level, path))
            else:
                print(f"WARNING: missing {path}, skipping")

    rows = []
    for method, level, path in runs:
        model, ckpt = load_model(path)
        results = measure(model, loaders, device, eval_test=True)

        if method == "none":
            compression = "None"
            note_head = f"Dense 1D CNN baseline ({BASELINE_CHECKPOINT.name}) re-measured for pruning comparison"
        else:
            compression = f"pruning: {method} {level}%"
            note_head = (
                f"Pruned {BASELINE_CHECKPOINT.name}, fine-tuned "
                f"{ckpt['finetune_epochs']} epochs at lr={ckpt['finetune_lr']:g}"
            )

        rows.append({
            "date": DATE,
            "author": AUTHOR,
            "model": "1D CNN (BearingCNN)",
            "input_repr": "raw waveform (1x5000)",
            "standardization": "Global z-score",
            "augmentation": "None",
            "compression": compression,
            "accuracy_test": results["test_accuracy"],
            "model_size_mb": results["size_mb"],
            "latency_ms_per_sample": results["latency"]["mean_ms"],
            "notes": (
                f"{note_head}. zipped_mb={results['zipped_size_mb']:.5f}; "
                f"val_acc={results['val_accuracy']:.4f}; {HARDWARE}"
            ),
        })

        print(
            f"{compression:<28} test_acc={results['test_accuracy']:.4f} "
            f"size={results['size_mb']:.5f}MB zipped={results['zipped_size_mb']:.5f}MB "
            f"latency={results['latency']['mean_ms']:.3f}ms"
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Log pruning results to experiments.csv")
    parser.add_argument("--save", action="store_true", help="Append the rows to results/experiments.csv")
    args = parser.parse_args()

    rows = collect_rows()

    df = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=False)
    next_id = 0 if df.empty else int(df.iloc[-1]["experiment_id"]) + 1

    for offset, row in enumerate(rows):
        df.loc[len(df)] = {"experiment_id": next_id + offset, **row}

    if args.save:
        df.to_csv(CSV_PATH, index=False)
        print(f"\nAppended {len(rows)} rows to {CSV_PATH} (ids {next_id}-{next_id + len(rows) - 1})")
    else:
        print(f"\nPreview only ({len(rows)} rows, ids {next_id}-{next_id + len(rows) - 1}). Re-run with --save to write.")


if __name__ == "__main__":
    main()
