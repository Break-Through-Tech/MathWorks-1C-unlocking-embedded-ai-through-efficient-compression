"""
src/cnn_1D/log_projection_results.py

Projection Part 2, step 3: final measurements, following
measurement-guide.md, logged into results/experiments.csv.

Re-measures the two dense source packages and every projected package
in ONE process, so the dense models and their projected versions are
timed in the same session:
    - accuracy : test split, scored here once (val accuracy goes in notes)
    - size     : the shipped package file, measure_model_size_mb(path=...)
    - latency  : batch size 1, one thread, shared defaults; like Gurbaj's
                 final benchmark, the timed call takes a raw window and
                 includes normalization and tensor conversion

Always writes results/projection_final.csv (all details, for the chart).
Only writes results/experiments.csv with --save, replacing this author's
earlier rows so the log holds one current set of projection results.

Run this in a Google Colab CPU runtime for the official latency numbers
(notebooks/08_projection_colab.ipynb). It records the platform in notes.

Usage (from the repository root):

    python -m src.cnn_1D.log_projection_results            # preview only
    python -m src.cnn_1D.log_projection_results --save     # write to CSV
"""

import argparse
import csv
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import load_split
from src.metrics import accuracy, measure_inference_latency_ms, measure_model_size_mb
from src.cnn_1D.cnn_model import CLASS_NAMES
from src.cnn_1D.cnn_project import RUN_INFO_NAME, load_model
from src.cnn_1D.projection_robustness import models_to_evaluate

CSV_PATH = Path("results/experiments.csv")
FINAL_CSV = Path("results/projection_final.csv")
AUTHOR = "Sabina Ruzieva"

# Same wording as Gurbaj's rows for the augmented original CNN.
AUGMENTATION_TEXT = {
    "none": "None",
    "shift_p0.5_noise_p0.5_20dB": "Time shift + Gaussian noise; p=0.5 each; 20 dB SNR",
}


def hardware_note():
    """Where the timing ran, e.g. 'Colab CPU' or 'Windows-11 CPU (not Colab ...)'."""
    try:
        import google.colab  # noqa: F401
        where = "Colab CPU"
    except ImportError:
        where = f"{platform.platform()} CPU (not Colab: re-run on Colab for official latency)"
    return f"{where}; 1 thread; batch=1; torch {torch.__version__}"


def standardize(X, checkpoint):
    mean = float(checkpoint["train_mean"])
    std = float(checkpoint["train_std"])
    return np.ascontiguousarray(((X - mean) / std)[:, None, :], dtype=np.float32)


def score(model, checkpoint, X, y):
    """Accuracy (shared metric) and cross-entropy on raw windows X."""
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    targets = np.array([label_map[str(label)] for label in y])
    with torch.inference_mode():
        logits = model(torch.from_numpy(standardize(X, checkpoint)))
    predictions = logits.argmax(dim=1).numpy()
    loss = float(F.cross_entropy(logits, torch.from_numpy(targets)))
    return accuracy(targets, predictions), loss


def time_model(model, checkpoint, X_pool):
    def predict_one(raw_sample):
        with torch.inference_mode():
            return model(torch.from_numpy(standardize(raw_sample, checkpoint))).argmax(dim=1).numpy()

    return measure_inference_latency_ms(predict_one, X_pool)


def collect():
    hardware = hardware_note()
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    final_rows, log_rows = [], []
    dense = {}

    for source, keep, path in models_to_evaluate():
        model, checkpoint = load_model(path)
        augmentation = checkpoint.get("augmentation", "none")
        label = "augmented" if augmentation != "none" else "clean"

        val_acc, val_loss = score(model, checkpoint, X_val, y_val)
        test_acc, test_loss = score(model, checkpoint, X_test, y_test)
        size_mb = measure_model_size_mb(path=path)
        latency = time_model(model, checkpoint, X_val)
        params = sum(p.numel() for p in model.parameters())

        if keep == 1.0:
            dense[source] = {"size_mb": size_mb, "params": params}
            info = None
        else:
            info = json.loads((path.parent / RUN_INFO_NAME).read_text(encoding="utf-8"))
            assert info["source_sha256"], path
            assert info["params"] == params, path

        ratio = dense[source]["size_mb"] / size_mb
        ranks = info["ranks"] if info else {}
        energy = {row["layer"]: row["energy_kept"] for row in info["summary"]} if info else {}

        final_rows.append({
            "model": label,
            "source": source.name,
            "keep": keep,
            "conv3_rank": ranks.get("features.8", 64),
            "conv4_rank": ranks.get("features.12", 96),
            "conv3_energy_kept": round(energy.get("features.8", 1.0), 4),
            "conv4_energy_kept": round(energy.get("features.12", 1.0), 4),
            "params": params,
            "size_mb": round(size_mb, 6),
            "compression_ratio": round(ratio, 3),
            "val_acc_before_finetune": info["val_before_finetune"]["accuracy"] if info else val_acc,
            "val_loss_before_finetune": info["val_before_finetune"]["loss"] if info else val_loss,
            "finetune_best_epoch": info["finetune_best_epoch"] if info else "",
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "latency_ms": round(latency["mean_ms"], 4),
            "latency_std_ms": round(latency["std_ms"], 4),
            "artifact": path.as_posix(),
            "hardware": hardware,
        })

        common = {
            "date": date.today().strftime("%m/%d/%Y"),
            "author": AUTHOR,
            "model": "1D CNN",
            "input_repr": "raw 5000-sample window",
            "standardization": "Global z-score",
            "augmentation": AUGMENTATION_TEXT[augmentation],
            "accuracy_test": test_acc,
            "model_size_mb": size_mb,
            "latency_ms_per_sample": latency["mean_ms"],
        }

        if info is None:
            notes = (
                f"Dense source for projection, re-timed in the same session as the "
                f"projected models. artifact={path.as_posix()} (Gurbaj's frozen package); "
                f"{params} params; val_acc={val_acc:.4f}; latency includes normalization "
                f"and tensor conversion; {hardware}"
            )
            log_rows.append({**common, "compression": "None", "notes": notes})
        else:
            notes = (
                f"Projection (truncated SVD) of conv3+conv4 of {source.name}, keeping "
                f"{keep:g} of each layer's rank (ranks {ranks['features.8']}/64 and "
                f"{ranks['features.12']}/96; energy kept {energy['features.8']:.2f}/"
                f"{energy['features.12']:.2f}); {params} params "
                f"({params / dense[source]['params']:.1%} of dense); fine-tuned "
                f"{info['finetune_epochs']} epochs at lr={info['finetune_lr']:g}, kept epoch "
                f"{info['finetune_best_epoch']} by val loss; val_acc before fine-tune="
                f"{info['val_before_finetune']['accuracy']:.4f}. artifact={path.as_posix()}; "
                f"saved as dense .pth inference package (weights + normalization + ranks; "
                f"load with cnn_project.load_model); {ratio:.2f}x smaller than the dense "
                f"package; val_acc={val_acc:.4f}; latency includes normalization and tensor "
                f"conversion; {hardware}"
            )
            log_rows.append({**common, "compression": "Projection", "notes": notes})

        print(
            f"{label:<9} keep={keep:<5} params={params:>6} size={size_mb:.6f}MB "
            f"({ratio:.2f}x) | test acc {test_acc:.4f} loss {test_loss:.5f} | "
            f"val acc {val_acc:.4f} | latency {latency['mean_ms']:.3f}ms"
        )

    return final_rows, log_rows


def main():
    parser = argparse.ArgumentParser(description="Log projection results to experiments.csv")
    parser.add_argument("--save", action="store_true",
                        help="Replace this author's rows in results/experiments.csv")
    args = parser.parse_args()

    torch.set_num_threads(1)
    final_rows, log_rows = collect()

    with FINAL_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"\nWrote {len(final_rows)} rows to {FINAL_CSV}")

    df = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=False)
    earlier = df["author"] == AUTHOR
    df = df[~earlier].reset_index(drop=True)
    next_id = 0 if df.empty else max(int(i) for i in df["experiment_id"]) + 1

    for offset, row in enumerate(log_rows):
        df.loc[len(df)] = {"experiment_id": next_id + offset, **row}

    summary = (
        f"{len(log_rows)} rows, ids {next_id}-{next_id + len(log_rows) - 1}; "
        f"replaces {int(earlier.sum())} earlier rows by {AUTHOR}"
    )
    if args.save:
        df.to_csv(CSV_PATH, index=False)
        print(f"Wrote {CSV_PATH} ({summary})")
    else:
        print(f"Preview only ({summary}). Re-run with --save to write.")


if __name__ == "__main__":
    main()
