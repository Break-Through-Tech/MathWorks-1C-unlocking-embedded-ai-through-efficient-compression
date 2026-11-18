"""Plot the projection Part 2 results: shipped size, test accuracy, and
noise robustness against how much detail (rank) each projected layer
keeps, for the clean and the augmented model.

Usage (from the repository root, after
python -m src.cnn_1D.projection_robustness and
python -m src.cnn_1D.log_projection_results):
    python results/projection_final_plot.py

Inputs:
    results/projection_final.csv
    results/projection_robustness.csv

Output:
    results/projection_final.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent
FINAL_CSV = RESULTS_DIR / "projection_final.csv"
ROBUSTNESS_CSV = RESULTS_DIR / "projection_robustness.csv"
OUT_PATH = RESULTS_DIR / "projection_final.png"

COLORS = {"clean": "tab:blue", "augmented": "tab:orange"}
SNR_STYLES = ((10, ":"), (5, "-"), (0, "--"))


def main():
    final = pd.read_csv(FINAL_CSV)
    robustness = pd.read_csv(ROBUSTNESS_CSV)

    # Evenly spaced positions, most detail on the left.
    levels = sorted(final["keep"].unique(), reverse=True)
    position = {keep: i for i, keep in enumerate(levels)}
    tick_labels = ["100%\n(original)" if k == 1.0 else f"{k:.0%}" for k in levels]

    fig, (ax_size, ax_acc, ax_noise) = plt.subplots(1, 3, figsize=(18, 5.5))

    sizes = final[final["model"] == "clean"].sort_values("keep", ascending=False)
    x = sizes["keep"].map(position)
    ax_size.bar(x, sizes["size_mb"], color="tab:green", alpha=0.8)
    for xi, mb, ratio, params in zip(x, sizes["size_mb"], sizes["compression_ratio"], sizes["params"]):
        ax_size.annotate(f"{ratio:.2f}x\n{params:,} params", (xi, mb), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8)
    ax_size.set_title("Shipped file size (same for clean and augmented)")
    ax_size.set_ylabel("MB (inference package on disk)")
    ax_size.set_ylim(0, sizes["size_mb"].max() * 1.25)

    for model, color in COLORS.items():
        rows = final[final["model"] == model].sort_values("keep", ascending=False)
        x = rows["keep"].map(position)
        ax_acc.plot(x, rows["test_accuracy"] * 100, "-o", color=color,
                    label=f"{model.capitalize()}: test, after fine-tune (102 windows)")
        ax_acc.plot(x, rows["val_acc_before_finetune"] * 100, "--o", color=color, alpha=0.7,
                    label=f"{model.capitalize()}: validation, before fine-tune (27 windows)")
    ax_acc.axhline(100 / 3, color="gray", linewidth=0.8, linestyle=":")
    ax_acc.text(0, 100 / 3 + 2, "chance (33%)", color="gray", fontsize=8)
    ax_acc.set_title("Accuracy: both models score 102/102 on test at every level")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=8, loc="lower left")

    noise = robustness[robustness["kind"] == "noise"]
    mean_acc = noise.groupby(["model", "keep", "snr_db"])["accuracy"].mean().reset_index()
    for model, color in COLORS.items():
        for snr, style in SNR_STYLES:
            rows = mean_acc[(mean_acc["model"] == model) & (mean_acc["snr_db"] == snr)]
            rows = rows.sort_values("keep", ascending=False)
            ax_noise.plot(rows["keep"].map(position), rows["accuracy"] * 100, style,
                          color=color, marker="o", label=f"{model.capitalize()}, {snr} dB SNR")
    ax_noise.set_title("Noisy validation windows (mean of 3 noise draws)")
    ax_noise.set_ylabel("Accuracy (%)")
    ax_noise.set_ylim(40, 103)
    ax_noise.legend(fontsize=8, loc="lower left", ncol=2)

    for ax in (ax_size, ax_acc, ax_noise):
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Rank kept in conv3 and conv4 (more compression ->)")
        ax.grid(alpha=0.3)

    fig.suptitle("Projection (SVD low-rank) on the final 1D CNN: Part 2", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
