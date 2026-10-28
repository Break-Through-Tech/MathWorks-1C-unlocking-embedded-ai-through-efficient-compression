"""
src/cnn_1D/cnn_prune.py

Pruning pipeline for the BearingCNN baseline (October Task #4).

Loads a trained checkpoint, removes a chosen fraction of the model, and
reports layer-by-layer sparsity plus validation accuracy before and
after pruning. Two pruning styles are supported:

    unstructured : global L1 pruning of individual weights, pooled
                   across all Conv1d and Linear layers. The least
                   important weights anywhere in the network are
                   removed first.
    structured   : L2 filter pruning per Conv1d layer — whole output
                   channels are zeroed, including their biases, so a
                   pruned filter is completely dead. The final
                   Linear(96, 3) is left untouched because removing
                   its output rows would remove entire classes.

Note on file size: pruning masks only ZERO weights; tensor shapes stay
the same, so the saved .pth file does not get smaller. That is expected
and is part of what this task has to demonstrate (the ZIPPED size
shrinks instead, because runs of zeros compress well).

Usage (from the repository root):

    python -m src.cnn_1D.cnn_prune --method unstructured --level 0.5
    python -m src.cnn_1D.cnn_prune --method structured --level 0.25
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_train import evaluate, prepare_data, set_seed

DEFAULT_CHECKPOINT = Path("cnn_outputs/cnn_rough_seed42.pth")
BATCH_SIZE = 32
SEED = 42
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4


def load_baseline(checkpoint_path, device):
    """Reload a trained BearingCNN and its checkpoint metadata."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = BearingCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def prunable_modules(model, method):
    """
    The (module, name) pairs each pruning style is allowed to touch.

    Unstructured pruning covers every Conv1d and Linear weight.
    Structured pruning covers only Conv1d weights: zeroing output rows
    of the final Linear(96, 3) would delete whole classes.
    """
    pairs = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            pairs.append((module, "weight"))
        elif isinstance(module, nn.Linear) and method == "unstructured":
            pairs.append((module, "weight"))
    return pairs


def apply_pruning(model, method, level):
    """
    Attach pruning masks that zero `level` (0-1) of the model.

    The masks stay attached so a later fine-tuning step keeps the
    pruned weights at zero while the surviving weights keep learning.
    """
    pairs = prunable_modules(model, method)

    if method == "unstructured":
        prune.global_unstructured(
            pairs,
            pruning_method=prune.L1Unstructured,
            amount=level
        )
    elif method == "structured":
        for module, name in pairs:
            prune.ln_structured(
                module, name=name, amount=level, n=2, dim=0
            )

            # ln_structured only masks the kernel weights. Also mask the
            # bias of each pruned filter, otherwise the "removed" channel
            # still emits a constant bias value.
            weight_mask = module.weight_mask
            dead_filters = (
                weight_mask.flatten(1).sum(dim=1) == 0
            )
            if module.bias is not None and dead_filters.any():
                bias_mask = (~dead_filters).to(module.bias.dtype)
                prune.custom_from_mask(module, "bias", bias_mask)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return model


def sparsity_report(model):
    """Print per-layer and overall weight sparsity; return overall."""
    total_weights = 0
    total_zeros = 0

    print(f"\n{'Layer':<22} {'Shape':<16} {'Zeros':>8} {'Total':>8} {'Sparsity':>9}")
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        weight = getattr(module, "weight")
        zeros = int((weight == 0).sum())
        count = weight.numel()
        total_weights += count
        total_zeros += zeros
        print(
            f"{module_name:<22} {str(tuple(weight.shape)):<16} "
            f"{zeros:>8} {count:>8} {zeros / count:>8.1%}"
        )

    overall = total_zeros / total_weights
    print(f"{'OVERALL (weights)':<22} {'':<16} {total_zeros:>8} {total_weights:>8} {overall:>8.1%}")
    return overall


def finetune(model, checkpoint, epochs, lr, device):
    """
    Briefly retrain the pruned model on the clean training split.

    The pruning masks stay attached, so pruned weights are pinned at
    zero while the surviving weights adapt. The epoch with the lowest
    validation loss is restored at the end — the same selection rule
    as the baseline training, and never based on test data.
    """
    mean = checkpoint["train_mean"]
    std = checkpoint["train_std"]

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        prepare_data(X_train, y_train, mean, std),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0
    )
    val_loader = DataLoader(
        prepare_data(X_val, y_val, mean, std),
        batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Fine-tune epoch {epoch:02d}/{epochs} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )

    model.load_state_dict(best_state)
    model.eval()
    print(f"Restored best fine-tune epoch: {best_epoch} (val loss {best_val_loss:.4f})")

    return model


def build_eval_loaders(checkpoint, include_test=False):
    """Standardized DataLoaders using the checkpoint's saved statistics."""
    mean = checkpoint["train_mean"]
    std = checkpoint["train_std"]

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    loaders = {
        "train": DataLoader(
            prepare_data(X_train, y_train, mean, std),
            batch_size=BATCH_SIZE, shuffle=False
        ),
        "val": DataLoader(
            prepare_data(X_val, y_val, mean, std),
            batch_size=BATCH_SIZE, shuffle=False
        ),
    }

    if include_test:
        X_test, y_test = load_split("test")
        loaders["test"] = DataLoader(
            prepare_data(X_test, y_test, mean, std),
            batch_size=BATCH_SIZE, shuffle=False
        )

    return loaders


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prune a trained BearingCNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"Baseline checkpoint to prune (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--method", choices=("unstructured", "structured"), required=True,
        help="unstructured = individual weights, structured = whole filters"
    )
    parser.add_argument(
        "--level", type=float, required=True,
        help="Fraction of the model to remove, e.g. 0.25, 0.5, 0.75, 0.9"
    )
    parser.add_argument(
        "--epochs", type=int, default=FINETUNE_EPOCHS,
        help=f"Fine-tuning epochs after pruning (default: {FINETUNE_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=FINETUNE_LR,
        help=f"Fine-tuning learning rate (default: {FINETUNE_LR})"
    )
    parser.add_argument(
        "--no-finetune", action="store_true",
        help="Skip the brief retraining step (for before/after comparisons)"
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.level < 1.0:
        parser.error("--level must be between 0 and 1 (exclusive)")

    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(SEED)

    device = torch.device("cpu")
    model, checkpoint = load_baseline(args.checkpoint, device)
    loaders = build_eval_loaders(checkpoint)

    print("Checkpoint:", args.checkpoint)
    print("Method:", args.method)
    print("Prune level:", args.level)

    val_loss, val_acc = evaluate(model, loaders["val"], device)
    print(f"\nBefore pruning | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    apply_pruning(model, args.method, args.level)
    sparsity_report(model)

    val_loss, val_acc = evaluate(model, loaders["val"], device)
    print(f"\nAfter pruning (no fine-tune) | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    if not args.no_finetune:
        print(f"\nFine-tuning for {args.epochs} epochs at lr={args.lr}")
        finetune(model, checkpoint, args.epochs, args.lr, device)

        val_loss, val_acc = evaluate(model, loaders["val"], device)
        print(f"\nAfter fine-tuning | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # Fine-tuning must not revive pruned weights.
        sparsity_report(model)


if __name__ == "__main__":
    main()
