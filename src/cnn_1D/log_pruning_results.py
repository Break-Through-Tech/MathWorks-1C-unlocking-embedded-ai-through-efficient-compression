"""
src/cnn_1D/log_pruning_results.py

Log the pruning experiments into results/experiments.csv, following
measurement-guide.md.

Re-measures the dense baseline and every pruned run under
cnn_outputs/pruned/ in ONE process, so latency for the baseline and
the pruned models comes from the same session (the guide asks for the
original to be re-timed alongside its compressed versions).

For each pruned run:
    - the shipped file is model_sparse.npz (nonzero weights + bit
      mask, see cnn_sparse.py); it is written if missing
    - size is that file, measured with measure_model_size_mb(path=...)
    - accuracy and latency come from the model loaded back FROM that
      file, so the logged numbers belong to what would be shipped
The dense baseline is logged with its normal .pth size.

Run this in a Google Colab CPU runtime for the official latency numbers
(see measurement-guide.md). It records the platform in `notes`.

Usage (from the repository root):

    python -m src.cnn_1D.log_pruning_results              # preview only
    python -m src.cnn_1D.log_pruning_results --save       # write to CSV

With --save, this author's earlier pruning rows are replaced, so the
CSV holds one current set of pruning measurements.
"""

import argparse
import platform
import re
from datetime import date
from pathlib import Path

import pandas as pd
import torch

from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_prune import build_eval_loaders, measure
from src.cnn_1D.cnn_sparse import load_sparse, save_sparse
from src.metrics import measure_model_size_mb

BASELINE_CHECKPOINT = Path("cnn_outputs/cnn_rough_seed42.pth")
PRUNED_DIR = Path("cnn_outputs/pruned")
CSV_PATH = Path("results/experiments.csv")
SPARSE_NAME = "model_sparse.npz"

AUTHOR = "Aigerim Zhadikbay"
RUN_DIR_RE = re.compile(r"(unstructured|structured)_(\d+(?:\.\d+)?)pct")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BearingCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def hardware_note():
    """Where the timing ran, e.g. 'Colab CPU' or 'macOS-15.6-arm64 CPU'."""
    try:
        import google.colab  # noqa: F401
        where = "Colab CPU"
    except ImportError:
        where = f"{platform.platform()} CPU (not Colab: re-run on Colab for official latency)"
    return f"{where}; 1 thread; batch=1; torch {torch.__version__}"


def pruned_runs(methods):
    """(method, level, run_dir) for every pruned run, in readable order."""
    runs = []
    for run_dir in PRUNED_DIR.iterdir():
        match = RUN_DIR_RE.fullmatch(run_dir.name)
        if match and match.group(1) in methods and (run_dir / "checkpoint.pth").exists():
            runs.append((match.group(1), float(match.group(2)), run_dir))
    order = {method: i for i, method in enumerate(methods)}
    return sorted(runs, key=lambda run: (order[run[0]], run[1]))


def collect_rows(methods):
    device = torch.device("cpu")
    hardware = hardware_note()
    today = date.today().strftime("%m/%d/%Y")

    baseline, baseline_ckpt = load_model(BASELINE_CHECKPOINT)
    loaders = build_eval_loaders(baseline_ckpt, include_test=True)

    common = {
        "date": today,
        "author": AUTHOR,
        "model": "1D CNN",
        "input_repr": "raw 5000-sample window",
        "standardization": "Global z-score",
        "augmentation": "None",
    }

    rows = []

    results = measure(baseline, loaders, device, eval_test=True)
    rows.append({
        **common,
        "compression": "None",
        "accuracy_test": results["test_accuracy"],
        "model_size_mb": measure_model_size_mb(path=BASELINE_CHECKPOINT),
        "latency_ms_per_sample": results["latency"]["mean_ms"],
        "notes": (
            f"Dense reference for pruning, re-timed in the same session as the pruned "
            f"models. artifact={BASELINE_CHECKPOINT} (same weights as "
            f"final_models/original_reference_seed42.pth); saved as dense .pth; "
            f"val_acc={results['val_accuracy']:.4f}; {hardware}"
        ),
    })
    print(f"{'baseline':<20} {_summary(rows[-1])}")

    for method, level, run_dir in pruned_runs(methods):
        checkpoint_path = run_dir / "checkpoint.pth"
        sparse_path = run_dir / SPARSE_NAME

        pruned, ckpt = load_model(checkpoint_path)
        if not sparse_path.exists():
            save_sparse(pruned, ckpt, sparse_path)

        # Measure the model as rebuilt from the shipped sparse file.
        shipped, _ = load_sparse(sparse_path)
        results = measure(shipped, loaders, device, eval_test=True)

        style = "individual weights, global L1" if method == "unstructured" else "whole conv filters, L2"
        rows.append({
            **common,
            "compression": "Pruning",
            "accuracy_test": results["test_accuracy"],
            "model_size_mb": measure_model_size_mb(path=sparse_path),
            "latency_ms_per_sample": results["latency"]["mean_ms"],
            "notes": (
                f"{method.capitalize()} pruning {level:g}% ({style}) of "
                f"{BASELINE_CHECKPOINT.name}, fine-tuned {ckpt['finetune_epochs']} epochs "
                f"at lr={ckpt['finetune_lr']:g}. artifact={sparse_path}; saved sparse "
                f"(nonzero float32 weights + 1-bit mask, np.savez_compressed); "
                f"dense .pth would be {measure_model_size_mb(path=checkpoint_path):.6f} MB; "
                f"timed after unpacking to a dense model; "
                f"val_acc={results['val_accuracy']:.4f}; {hardware}"
            ),
        })
        print(f"{run_dir.name:<20} {_summary(rows[-1])}")

    return rows


def _summary(row):
    return (
        f"test_acc={row['accuracy_test']:.4f} size={row['model_size_mb']:.6f}MB "
        f"latency={row['latency_ms_per_sample']:.3f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Log pruning results to experiments.csv")
    parser.add_argument(
        "--save", action="store_true",
        help="Replace this author's pruning rows in results/experiments.csv"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("unstructured", "structured"),
        default=["unstructured", "structured"], help="Pruning styles to log"
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    rows = collect_rows(args.methods)

    df = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=False)
    earlier = df["author"] == AUTHOR
    df = df[~earlier].reset_index(drop=True)
    next_id = 0 if df.empty else max(int(i) for i in df["experiment_id"]) + 1

    for offset, row in enumerate(rows):
        df.loc[len(df)] = {"experiment_id": next_id + offset, **row}

    summary = (
        f"{len(rows)} rows, ids {next_id}-{next_id + len(rows) - 1}; "
        f"replaces {int(earlier.sum())} earlier rows by {AUTHOR}"
    )
    if args.save:
        df.to_csv(CSV_PATH, index=False)
        print(f"\nWrote {CSV_PATH} ({summary})")
    else:
        print(f"\nPreview only ({summary}). Re-run with --save to write.")


if __name__ == "__main__":
    main()
