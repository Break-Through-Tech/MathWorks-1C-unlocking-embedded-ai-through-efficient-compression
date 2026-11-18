"""
src/cnn_1D/projection_build.py

Projection Part 2, step 1: build and save the final projected models.

Projects Gurbaj's two frozen original-width packages (clean reference and
shift+noise augmented) at each detail level, fine-tunes briefly with the
same settings as Part 1, and saves every result as an inference package
under cnn_outputs/projected/<source>/rank_<level>pct/ (model.pth and
run_info.json).

Only the training and validation splits are used here. The test split is
scored once, later, by log_projection_results.py.

Usage (from the repository root):

    python -m src.cnn_1D.projection_build
    python -m src.cnn_1D.projection_build --overwrite   # rebuild everything
"""

import argparse
import hashlib
import platform
from pathlib import Path

import torch

from src.cnn_1D.cnn_project import (
    DEFAULT_LAYERS, DEFAULT_OUTPUT_DIR, FINETUNE_EPOCHS, FINETUNE_LR, SEED,
    build_eval_loaders, finetune, load_model, project_model, run_dir_for,
    save_projected_package,
)
from src.cnn_1D.cnn_train import evaluate, set_seed

SOURCES = (
    Path("cnn_outputs/final_models/original_reference_seed42.pth"),
    Path("cnn_outputs/augmentation_models/original_both_seed42.pth"),
)
KEEP_LEVELS = (0.5, 0.25, 0.1, 0.05, 0.02)
MODE = "rank"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the final projected models.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace packages that already exist")
    args = parser.parse_args()

    device = torch.device("cpu")

    for source in SOURCES:
        model, checkpoint = load_model(source, device)
        loaders = build_eval_loaders(checkpoint)
        dense_loss, dense_acc = evaluate(model, loaders["val"], device)
        dense_params = sum(p.numel() for p in model.parameters())

        print(f"\n=== {source} | augmentation={checkpoint.get('augmentation', 'none')} "
              f"| val acc {dense_acc:.2%} | val loss {dense_loss:.6f}")

        for keep in KEEP_LEVELS:
            run_dir = run_dir_for(DEFAULT_OUTPUT_DIR, source, MODE, keep)
            if (run_dir / "model.pth").exists() and not args.overwrite:
                print(f"keep={keep:<5} exists, skipped ({run_dir})")
                continue

            projected, ranks, summary = project_model(model, DEFAULT_LAYERS, keep, MODE)
            before_loss, before_acc = evaluate(projected, loaders["val"], device)

            set_seed(SEED)
            best_epoch = finetune(projected, checkpoint, loaders["val"],
                                  FINETUNE_EPOCHS, FINETUNE_LR, device)
            after_loss, after_acc = evaluate(projected, loaders["val"], device)
            params = sum(p.numel() for p in projected.parameters())

            run_info = {
                "source_checkpoint": source.as_posix(),
                "source_sha256": sha256(source),
                "augmentation": checkpoint.get("augmentation", "none"),
                "layers": list(DEFAULT_LAYERS),
                "mode": MODE,
                "keep": keep,
                "ranks": ranks,
                "summary": summary,
                "params_dense": dense_params,
                "params": params,
                "val_dense": {"loss": dense_loss, "accuracy": dense_acc},
                "val_before_finetune": {"loss": before_loss, "accuracy": before_acc},
                "val_after_finetune": {"loss": after_loss, "accuracy": after_acc},
                "finetune_epochs": FINETUNE_EPOCHS,
                "finetune_lr": FINETUNE_LR,
                "finetune_best_epoch": best_epoch,
                "seed": SEED,
                "test_split_used": False,
                "torch_version": str(torch.__version__),
                "platform": platform.platform(),
            }
            path = save_projected_package(projected, checkpoint, ranks, run_dir,
                                          run_info, overwrite=args.overwrite)

            print(
                f"keep={keep:<5} ranks={list(ranks.values())} params={params:>6} | "
                f"val acc {before_acc:.2%} -> {after_acc:.2%} | "
                f"val loss {before_loss:.4f} -> {after_loss:.6f} | {path}"
            )


if __name__ == "__main__":
    main()
