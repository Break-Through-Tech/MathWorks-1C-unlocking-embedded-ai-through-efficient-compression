"""Plot the pruning experiment results from experiments.csv:
test accuracy and model size (shipped sparse file vs dense .pth) against the fraction of
weights removed, one line per pruning style.

Usage:
    python results/pruning_plot.py               # reads results/experiments.csv
    python results/pruning_plot.py path/to/other.csv

Output:
    results/pruning_accuracy_vs_level.png
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent
DEFAULT_CSV = RESULTS_DIR / "experiments.csv"
OUT_PATH = RESULTS_DIR / "pruning_accuracy_vs_level.png"

# Colorblind-safe pair (validated: CVD ΔE 24.7, normal ΔE 33.6)
COLORS = {"unstructured": "#2a78d6", "structured": "#eb6834"}
CHANCE_ACCURACY = 1 / 3  # three balanced classes

PRUNING_RE = re.compile(r"(Unstructured|Structured) pruning (\d+(?:\.\d+)?)%")
DENSE_PTH_RE = re.compile(r"dense \.pth would be ([0-9.]+) MB")
REFERENCE_NOTE = "Dense reference for pruning"


def load_pruning_rows(csv_path: Path) -> pd.DataFrame:
    """The dense reference row plus every pruning row, with the method,
    prune level, and dense .pth size parsed out of `notes`."""
    df = pd.read_csv(csv_path, dtype={"compression": str, "notes": str})

    rows = []
    for _, row in df.iterrows():
        notes = str(row["notes"])
        pruning = PRUNING_RE.match(notes)

        if pruning and str(row["compression"]).strip() == "Pruning":
            method, level = pruning.group(1).lower(), float(pruning.group(2))
            dense_mb = float(DENSE_PTH_RE.search(notes).group(1))
        elif notes.startswith(REFERENCE_NOTE):
            method, level = "baseline", 0.0
            dense_mb = float(row["model_size_mb"])
        else:
            continue  # not one of the pruning rows

        rows.append({
            "method": method,
            "level": level,
            "accuracy_test": float(row["accuracy_test"]),
            "size_mb": float(row["model_size_mb"]),
            "dense_mb": dense_mb,
        })

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_acc, ax_size) = plt.subplots(1, 2, figsize=(11, 4.5))

    baseline = df[df["method"] == "baseline"]

    for method in ("unstructured", "structured"):
        sub = df[df["method"] == method].sort_values("level")
        if sub.empty:
            continue
        # Anchor both lines at 0% removed = the dense baseline.
        levels = [0] + sub["level"].tolist()
        acc = baseline["accuracy_test"].tolist() + sub["accuracy_test"].tolist()
        shipped = baseline["size_mb"].tolist() + sub["size_mb"].tolist()
        color = COLORS[method]

        ax_acc.plot(levels, acc, marker="o", markersize=6, linewidth=2,
                    color=color, label=method, zorder=3)
        ax_acc.annotate(method, (levels[-1], acc[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        color="#52514e", fontsize=9, va="center")

        ax_size.plot(levels, shipped, marker="o", markersize=6, linewidth=2,
                     color=color, label=f"sparse file ({method})", zorder=3)

    ax_acc.axhline(CHANCE_ACCURACY, linestyle=":", color="#9a9a94", linewidth=1.5, zorder=1)
    ax_acc.annotate("chance (3 classes)", (0, CHANCE_ACCURACY),
                    textcoords="offset points", xytext=(4, 5),
                    color="#52514e", fontsize=8)
    ax_acc.set_xlabel("Weights removed (%)")
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_title("Accuracy vs. pruning level")
    ax_acc.set_ylim(0.25, 1.05)
    ax_acc.set_xlim(-3, 108)

    # A dense .pth never shrinks — plot it flat for the contrast.
    sizes = df["dense_mb"]
    ax_size.axhline(sizes.iloc[0], linestyle="--", color="#9a9a94", linewidth=1.5, zorder=2)
    ax_size.annotate("dense .pth (unchanged)", (0, sizes.iloc[0]),
                     textcoords="offset points", xytext=(4, -12),
                     color="#52514e", fontsize=8)
    ax_size.set_xlabel("Weights removed (%)")
    ax_size.set_ylabel("Model size (MB)")
    ax_size.set_title("File size vs. pruning level")
    ax_size.set_ylim(0, sizes.iloc[0] * 1.15)

    for ax in (ax_acc, ax_size):
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("1D CNN pruning: a dense .pth never shrinks, a sparse file does", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = load_pruning_rows(csv_path)

    if df[df["method"] != "baseline"].empty:
        sys.exit(f"No pruning rows found in {csv_path} — run the pruning grid and log it first.")

    plot(df, OUT_PATH)


if __name__ == "__main__":
    main()
