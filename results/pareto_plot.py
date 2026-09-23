"""Plot accuracy vs. model size and accuracy vs. latency from experiments.csv,
highlighting the Pareto frontier (models where you can't gain accuracy
without paying more size/latency).

Usage:
    python results/pareto_plot.py                # reads results/experiments.csv
    python results/pareto_plot.py path/to/other.csv

Output:
    results/pareto_accuracy_vs_size.png
    results/pareto_accuracy_vs_latency.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent
DEFAULT_CSV = RESULTS_DIR / "experiments.csv"

REQUIRED_COLUMNS = ["experiment_id", "accuracy_test", "model_size_mb", "latency_ms_per_sample"]


def pareto_frontier(df: pd.DataFrame, cost_col: str) -> pd.DataFrame:
    """Rows not dominated by any other row: no other experiment has both
    higher-or-equal accuracy AND lower-or-equal cost (with at least one strict).

    Returns the frontier sorted by cost, ready to draw as a step line.
    """
    df = df.sort_values([cost_col, "accuracy_test"], ascending=[True, False])
    frontier_rows = []
    best_acc = -float("inf")
    for _, row in df.iterrows():
        if row["accuracy_test"] > best_acc:
            frontier_rows.append(row)
            best_acc = row["accuracy_test"]
    return pd.DataFrame(frontier_rows)


def plot_pareto(df: pd.DataFrame, cost_col: str, cost_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # One color per compression method so baselines vs. compressed variants are easy to tell apart
    compression = df["compression"].fillna("none").replace("", "none")
    for method, group in df.groupby(compression):
        ax.scatter(group[cost_col], group["accuracy_test"], label=method, s=60, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(
            str(row["experiment_id"]),
            (row[cost_col], row["accuracy_test"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    frontier = pareto_frontier(df, cost_col)
    ax.plot(
        frontier[cost_col],
        frontier["accuracy_test"],
        linestyle="--",
        color="gray",
        drawstyle="steps-post",
        label="Pareto frontier",
        zorder=2,
    )

    ax.set_xlabel(cost_label)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy vs. {cost_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Compression")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"{csv_path} is missing expected columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        print(f"No completed experiments in {csv_path} yet — nothing to plot.")
        print("Log a row (see results/experiments.csv header) and rerun.")
        return

    plot_pareto(df, "model_size_mb", "Model size (MB)", RESULTS_DIR / "pareto_accuracy_vs_size.png")
    plot_pareto(
        df,
        "latency_ms_per_sample",
        "Latency (ms/sample)",
        RESULTS_DIR / "pareto_accuracy_vs_latency.png",
    )


if __name__ == "__main__":
    main()
