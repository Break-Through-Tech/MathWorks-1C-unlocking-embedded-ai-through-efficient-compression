"""Plot the projection Part 1 sweep from results/projection_sweep.csv:
validation accuracy, validation loss, model size, and latency against
how much detail (rank) each projected layer keeps.

Usage (from the repository root, after python -m src.cnn_1D.projection_sweep):
    python results/projection_plot.py

Output:
    results/projection_sweep.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent
CSV_PATH = RESULTS_DIR / "projection_sweep.csv"
OUT_PATH = RESULTS_DIR / "projection_sweep.png"

LINES = (
    ("clean", False, "Clean model, no fine-tune", "tab:blue", "--"),
    ("clean", True, "Clean model, fine-tuned", "tab:blue", "-"),
    ("augmented", False, "Augmented model, no fine-tune", "tab:orange", "--"),
    ("augmented", True, "Augmented model, fine-tuned", "tab:orange", "-"),
)


def with_baseline(df, model, finetuned):
    """Rows for one line, with the unprojected baseline as its first point."""
    baseline = df[(df["model"] == model) & (df["keep"] == 1.0)]
    projected = df[(df["model"] == model) & (df["keep"] < 1.0) & (df["finetuned"] == finetuned)]
    return pd.concat([baseline, projected]).sort_values("keep", ascending=False)


def main():
    df = pd.read_csv(CSV_PATH)
    df["finetuned"] = df["finetuned"].astype(str) == "True"

    # Evenly spaced positions, most detail on the left.
    levels = sorted(df["keep"].unique(), reverse=True)
    position = {keep: i for i, keep in enumerate(levels)}
    tick_labels = ["100%\n(original)" if k == 1.0 else f"{k:.0%}" for k in levels]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_acc, ax_loss), (ax_size, ax_lat) = axes

    for model, finetuned, label, color, style in LINES:
        rows = with_baseline(df, model, finetuned)
        x = rows["keep"].map(position)
        ax_acc.plot(x, rows["val_accuracy"] * 100, style, color=color, marker="o", label=label)
        ax_loss.plot(x, rows["val_loss"], style, color=color, marker="o", label=label)

    ax_acc.set_title("Validation accuracy (27 windows)")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.axhline(100 / 3, color="gray", linewidth=0.8, linestyle=":")
    ax_acc.text(0, 100 / 3 + 2, "chance (33%)", color="gray", fontsize=8)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=8, loc="lower left")

    ax_loss.set_title("Validation loss (lower is better; shows damage accuracy hides)")
    ax_loss.set_ylabel("Cross-entropy loss (log scale)")
    ax_loss.set_yscale("log")

    sizes = with_baseline(df, "clean", True)
    x = sizes["keep"].map(position)
    ax_size.plot(x, sizes["size_mb"], "-o", color="tab:green", label=".pth file")
    ax_size.plot(x, sizes["zipped_size_mb"], "--o", color="tab:green", label="zipped .pth")
    for xi, mb, params in zip(x, sizes["size_mb"], sizes["params"]):
        ax_size.annotate(f"{params:,} params", (xi, mb), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    ax_size.set_title("Model size (same for clean and augmented)")
    ax_size.set_ylabel("MB")
    ax_size.set_ylim(0, sizes["size_mb"].max() * 1.2)
    ax_size.legend(fontsize=8)

    for model, color in (("clean", "tab:blue"), ("augmented", "tab:orange")):
        rows = with_baseline(df, model, True)
        ax_lat.plot(rows["keep"].map(position), rows["latency_ms"], "-o",
                    color=color, label=f"{model.capitalize()} model, fine-tuned")
    ax_lat.set_title("Latency at batch size 1, one CPU thread")
    ax_lat.set_ylabel("ms per sample")
    ax_lat.set_ylim(0, df["latency_ms"].max() * 1.3)
    ax_lat.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Rank kept in conv3 and conv4 (less detail, more compression ->)")
        ax.grid(alpha=0.3)

    fig.suptitle("Projection (SVD low-rank) on the rough 1D CNN: Part 1, validation only",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
