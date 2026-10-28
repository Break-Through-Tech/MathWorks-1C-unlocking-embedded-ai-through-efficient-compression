"""Plot the pruning experiment results from experiments.csv:
test accuracy and model size (raw vs zipped) against the fraction of
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

COMPRESSION_RE = re.compile(r"pruning: (unstructured|structured) (\d+)%")
ZIPPED_RE = re.compile(r"zipped_mb=([0-9.]+)")


def load_pruning_rows(csv_path: Path) -> pd.DataFrame:
    """The dense-CNN baseline row plus every pruning row, with the
    method, prune level, and zipped size parsed out."""
    df = pd.read_csv(csv_path, dtype={"compression": str, "notes": str})

    rows = []
    for _, row in df.iterrows():
        zipped = ZIPPED_RE.search(str(row["notes"]))
        if zipped is None:
            continue  # not one of the CNN measurement rows

        match = COMPRESSION_RE.fullmatch(str(row["compression"]).strip())
        if match:
            method, level = match.group(1), int(match.group(2))
        elif str(row["compression"]).strip() in ("None", "", "nan"):
            method, level = "baseline", 0
        else:
            continue  # some other compression technique

        rows.append({
            "method": method,
            "level": level,
            "accuracy_test": float(row["accuracy_test"]),
            "size_mb": float(row["model_size_mb"]),
            "zipped_mb": float(zipped.group(1)),
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
        zipped = baseline["zipped_mb"].tolist() + sub["zipped_mb"].tolist()
        color = COLORS[method]

        ax_acc.plot(levels, acc, marker="o", markersize=6, linewidth=2,
                    color=color, label=method, zorder=3)
        ax_acc.annotate(method, (levels[-1], acc[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        color="#52514e", fontsize=9, va="center")

        ax_size.plot(levels, zipped, marker="o", markersize=6, linewidth=2,
                     color=color, label=f"zipped ({method})", zorder=3)

    ax_acc.axhline(CHANCE_ACCURACY, linestyle=":", color="#9a9a94", linewidth=1.5, zorder=1)
    ax_acc.annotate("chance (3 classes)", (0, CHANCE_ACCURACY),
                    textcoords="offset points", xytext=(4, 5),
                    color="#52514e", fontsize=8)
    ax_acc.set_xlabel("Weights removed (%)")
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_title("Accuracy vs. pruning level")
    ax_acc.set_ylim(0.25, 1.05)
    ax_acc.set_xlim(-3, 108)

    # The raw .pth never shrinks — plot it flat for the contrast.
    sizes = df["size_mb"]
    ax_size.axhline(sizes.iloc[0], linestyle="--", color="#9a9a94", linewidth=1.5, zorder=2)
    ax_size.annotate("raw .pth (unchanged)", (0, sizes.iloc[0]),
                     textcoords="offset points", xytext=(4, -12),
                     color="#52514e", fontsize=8)
    ax_size.set_xlabel("Weights removed (%)")
    ax_size.set_ylabel("Model size (MB)")
    ax_size.set_title("File size vs. pruning level")
    ax_size.set_ylim(0, sizes.iloc[0] * 1.15)

    for ax in (ax_acc, ax_size):
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("1D CNN pruning: masks zero weights but only the zipped file shrinks", y=1.02)
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
