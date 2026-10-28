"""
src/cnn_1D/cnn_project.py

Projection (low-rank factorization) pipeline for the BearingCNN baseline
(October compression task: projection).

Each selected Conv1d layer is replaced by two thinner layers that do
almost the same job:

    Conv1d(in, out, k)  ->  Conv1d(in, r, k, no bias) -> Conv1d(r, out, 1)

The new weights come from a truncated SVD of the layer's weight matrix
(shape out x in*k). At full rank the two layers reproduce the original
layer exactly; lower ranks drop the least important directions first.

Unlike pruning, the replacement layers are physically smaller, so the
saved file and the amount of computation both shrink. A layer only gets
smaller below its break-even rank, out*in*k / (in*k + out). Above that,
the two thin layers together hold MORE weights than the original.

How much "detail" to keep can be chosen two ways:

    rank   : keep this fraction of each layer's full rank (e.g. 0.5)
    energy : keep the smallest rank whose singular values hold at least
             this fraction of the layer's squared singular-value energy

Usage (from the repository root):

    python -m src.cnn_1D.cnn_project --report
    python -m src.cnn_1D.cnn_project --keep 0.5
    python -m src.cnn_1D.cnn_project --keep 0.9 --mode energy
    python -m src.cnn_1D.cnn_project --keep 0.25 \\
        --checkpoint cnn_outputs/cnn_rough_aug_seed42.pth
"""

import argparse
import copy
import tempfile
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import load_split
from src.cnn_1D.cnn_augmentation import AugmentedTrainingDataset
from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_train import evaluate, prepare_data, set_seed
from src.metrics import measure_inference_latency_ms, measure_model_size_mb

DEFAULT_CHECKPOINT = Path("cnn_outputs/cnn_rough_seed42.pth")
DEFAULT_OUTPUT_DIR = Path("cnn_outputs/projected")

# Conv blocks 3 and 4 hold ~89% of the model's weights (see --report).
DEFAULT_LAYERS = ("features.8", "features.12")

# The only augmentation AugmentedTrainingDataset implements.
AUGMENTED_LABEL = "shift_p0.5_noise_p0.5_20dB"

BATCH_SIZE = 32
SEED = 42
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4


# ---------------------------------------------------------------------------
# Layer sizes and ranks
# ---------------------------------------------------------------------------

def _matrix_shape(weight):
    """A layer's weight viewed as a matrix: (outputs, inputs per output)."""
    return weight.shape[0], weight[0].numel()


def factorized_weight_count(weight, rank):
    """Weights in the two thin layers that replace `weight` at `rank`."""
    rows, cols = _matrix_shape(weight)
    return rank * (rows + cols)


def break_even_rank(weight):
    """Largest rank at which the two thin layers are still smaller."""
    rows, cols = _matrix_shape(weight)
    return (rows * cols - 1) // (rows + cols)


def layer_report(model):
    """Conv1d/Linear layers sorted by parameter count, largest first."""
    total = sum(p.numel() for p in model.parameters())

    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        params = sum(p.numel() for p in module.parameters())
        rows.append({
            "name": name,
            "shape": tuple(module.weight.shape),
            "params": params,
            "share": params / total,
            "full_rank": min(_matrix_shape(module.weight)),
            "break_even_rank": break_even_rank(module.weight),
        })

    rows.sort(key=lambda row: row["params"], reverse=True)
    return rows, total


def print_layer_report(model):
    rows, total = layer_report(model)

    print(
        f"\n{'Layer':<14} {'Weight shape':<14} {'Params':>8} {'Share':>7} "
        f"{'Full rank':>10} {'Break-even':>11}"
    )
    for row in rows:
        print(
            f"{row['name']:<14} {str(row['shape']):<14} {row['params']:>8} "
            f"{row['share']:>7.1%} {row['full_rank']:>10} {row['break_even_rank']:>11}"
        )
    print(f"{'TOTAL (all parameters)':<37} {total:>8}")


def conv_svd(conv):
    """SVD of a Conv1d weight reshaped to (out, in*k), in float64."""
    matrix = conv.weight.detach().double().reshape(conv.out_channels, -1)
    return torch.linalg.svd(matrix, full_matrices=False)


def select_rank(singular_values, keep, mode):
    """
    Rank to keep for one layer.

    mode="rank"   : round(keep * full rank), at least 1
    mode="energy" : smallest rank whose squared singular values reach
                    `keep` of the layer's total
    """
    full_rank = len(singular_values)

    if mode == "rank":
        return min(full_rank, max(1, round(keep * full_rank)))

    if mode == "energy":
        energy = singular_values.pow(2).cumsum(0)
        energy = energy / energy[-1]
        target = torch.tensor(keep, dtype=energy.dtype)
        return min(full_rank, int(torch.searchsorted(energy, target)) + 1)

    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------

def _factorized_shell(conv, rank):
    """The two thin layers that replace `conv`, with untrained weights."""
    if conv.groups != 1 or conv.padding_mode != "zeros":
        raise ValueError("Only plain Conv1d layers (groups=1, zero padding) are supported.")

    return nn.Sequential(
        nn.Conv1d(
            conv.in_channels, rank, conv.kernel_size,
            stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, bias=False
        ),
        nn.Conv1d(rank, conv.out_channels, 1, bias=conv.bias is not None),
    )


def factorize_conv(conv, rank):
    """
    Split one Conv1d into Conv1d(in, rank, k) -> Conv1d(rank, out, 1).

    With W = U S Vh, the first layer gets sqrt(S) Vh and the second
    gets U sqrt(S), so neither layer starts out much larger in scale
    than the other when fine-tuning begins.
    """
    U, S, Vh = conv_svd(conv)
    if not 1 <= rank <= len(S):
        raise ValueError(f"rank must be between 1 and {len(S)}, got {rank}")

    root = S[:rank].sqrt()
    shell = _factorized_shell(conv, rank).to(conv.weight.device)

    with torch.no_grad():
        shell[0].weight.copy_(
            (root[:, None] * Vh[:rank]).reshape(shell[0].weight.shape)
        )
        shell[1].weight.copy_(
            (U[:, :rank] * root).reshape(shell[1].weight.shape)
        )
        if conv.bias is not None:
            shell[1].bias.copy_(conv.bias)

    return shell


def _replace_module(model, name, new_module):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new_module)


def project_model(model, layers=DEFAULT_LAYERS, keep=0.5, mode="rank"):
    """
    Return a projected copy of `model`, the rank chosen for each layer,
    and a per-layer summary. The original model is left untouched.
    """
    projected = copy.deepcopy(model)
    ranks = {}
    summary = []

    for name in layers:
        conv = projected.get_submodule(name)
        if not isinstance(conv, nn.Conv1d):
            raise TypeError(f"{name} is {type(conv).__name__}, expected Conv1d")

        _, S, _ = conv_svd(conv)
        rank = select_rank(S, keep, mode)
        energy = S.pow(2)

        summary.append({
            "layer": name,
            "full_rank": len(S),
            "rank": rank,
            "break_even_rank": break_even_rank(conv.weight),
            "energy_kept": float(energy[:rank].sum() / energy.sum()),
            "params_before": sum(p.numel() for p in conv.parameters()),
            "params_after": (
                factorized_weight_count(conv.weight, rank)
                + (conv.out_channels if conv.bias is not None else 0)
            ),
        })

        _replace_module(projected, name, factorize_conv(conv, rank))
        ranks[name] = rank

    return projected, ranks, summary


def print_projection_summary(summary):
    print(
        f"\n{'Layer':<14} {'Rank':>10} {'Break-even':>11} "
        f"{'Energy kept':>12} {'Params':>18}"
    )
    for row in summary:
        rank = f"{row['rank']}/{row['full_rank']}"
        params = f"{row['params_before']} -> {row['params_after']}"
        print(
            f"{row['layer']:<14} {rank:>10} {row['break_even_rank']:>11} "
            f"{row['energy_kept']:>12.2%} {params:>18}"
        )
        if row["rank"] > row["break_even_rank"]:
            print(f"  WARNING: {row['layer']} is above break-even and got LARGER.")


def build_projected_model(ranks):
    """
    An untrained BearingCNN with the layers in `ranks` already split,
    ready for load_state_dict(). An empty `ranks` gives the plain model.
    """
    model = BearingCNN()
    for name, rank in ranks.items():
        _replace_module(
            model, name, _factorized_shell(model.get_submodule(name), rank)
        )
    return model


def load_model(checkpoint_path, device=torch.device("cpu")):
    """Reload a baseline or projected checkpoint and its metadata."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = build_projected_model(checkpoint.get("projection_ranks", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------------
# Data, fine-tuning, measurement
# ---------------------------------------------------------------------------

def build_eval_loaders(checkpoint, include_test=False):
    """Standardized val (and optionally test) loaders, using the checkpoint's statistics."""
    mean = checkpoint["train_mean"]
    std = checkpoint["train_std"]
    splits = ("val", "test") if include_test else ("val",)

    loaders = {}
    for split in splits:
        X, y = load_split(split)
        loaders[split] = DataLoader(
            prepare_data(X, y, mean, std),
            batch_size=BATCH_SIZE, shuffle=False
        )
    return loaders


def build_train_dataset(checkpoint):
    """
    Training data for fine-tuning, matching how the checkpoint was trained:
    an augmented checkpoint keeps its augmentation, a clean one stays clean.
    """
    mean = checkpoint["train_mean"]
    std = checkpoint["train_std"]
    augmentation = checkpoint.get("augmentation", "none")

    X_train, y_train = load_split("train")

    if augmentation == "none":
        return prepare_data(X_train, y_train, mean, std)
    if augmentation == AUGMENTED_LABEL:
        return AugmentedTrainingDataset(X_train, y_train, mean, std, seed=SEED)

    raise ValueError(f"No training dataset for augmentation {augmentation!r}")


def finetune(model, checkpoint, val_loader, epochs, lr, device):
    """
    Briefly retrain the projected model. The epoch with the lowest
    validation loss is kept, and the projected model before any
    fine-tuning counts as epoch 0, so fine-tuning can never make the
    selected model worse on validation loss. Test data is never used.
    """
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        build_train_dataset(checkpoint),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, _ = evaluate(model, val_loader, device)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
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
    print(f"Kept fine-tune epoch: {best_epoch} (val loss {best_val_loss:.4f})")

    return best_epoch


def measure(model, loaders, device):
    """
    Parameter count, val (and test, if loaded) accuracy, .pth size,
    zipped .pth size, and ms/sample latency at batch size 1 on a single
    CPU thread, through the team's shared src/metrics.py.
    """
    model.eval()
    results = {"params": sum(p.numel() for p in model.parameters())}

    results["val_loss"], results["val_accuracy"] = evaluate(model, loaders["val"], device)
    if "test" in loaders:
        _, results["test_accuracy"] = evaluate(model, loaders["test"], device)

    results["size_mb"] = measure_model_size_mb(model=model, kind="pytorch")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "model.pth"
        torch.save(model.state_dict(), raw_path)
        zip_path = Path(tmp) / "model.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(raw_path, arcname="model.pth")
        results["zipped_size_mb"] = zip_path.stat().st_size / 1_000_000

    X_pool = loaders["val"].dataset.tensors[0].numpy()

    def predict_fn(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x)).argmax(dim=1)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        results["latency"] = measure_inference_latency_ms(predict_fn, X_pool)
    finally:
        torch.set_num_threads(threads)

    return results


def print_results(label, results):
    line = (
        f"{label:<12} | params {results['params']:>6} | "
        f"val acc {results['val_accuracy']:.2%} | val loss {results['val_loss']:.4f} | "
    )
    if "test_accuracy" in results:
        line += f"test acc {results['test_accuracy']:.2%} | "
    line += (
        f"size {results['size_mb']:.4f} MB | zipped {results['zipped_size_mb']:.4f} MB | "
        f"latency {results['latency']['mean_ms']:.3f} ms"
    )
    print(line)


def save_projected_checkpoint(model, checkpoint, args, ranks, summary,
                              best_epoch, results):
    """
    Save in the baseline's checkpoint format plus the projection settings.
    load_model() rebuilds the split layers from "projection_ranks".
    """
    run_dir = (
        args.output_dir / args.checkpoint.stem
        / f"{args.mode}_{args.keep * 100:g}pct"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "checkpoint.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "projection_ranks": ranks,
        "projection_mode": args.mode,
        "projection_keep": args.keep,
        "projection_summary": summary,
        "seed": SEED,
        "train_mean": checkpoint["train_mean"],
        "train_std": checkpoint["train_std"],
        "class_names": checkpoint["class_names"],
        "augmentation": checkpoint.get("augmentation", "none"),
        "torch_version": str(torch.__version__),
        "finetune_epochs": 0 if args.no_finetune else args.epochs,
        "finetune_lr": args.lr,
        "finetune_best_epoch": best_epoch,
        "source_checkpoint": str(args.checkpoint),
        "results": results,
    }, out_path)

    return out_path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def print_spectrum_report(model, layers):
    """For each layer: ranks needed to keep common energy levels."""
    print(f"\n{'Layer':<14} {'Full rank':>10} {'Break-even':>11}   Rank needed for energy kept")
    for name in layers:
        conv = model.get_submodule(name)
        _, S, _ = conv_svd(conv)
        needed = ", ".join(
            f"{level:.0%}: {select_rank(S, level, 'energy')}"
            for level in (0.5, 0.75, 0.9, 0.95, 0.99)
        )
        print(f"{name:<14} {len(S):>10} {break_even_rank(conv.weight):>11}   {needed}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Project (low-rank factorize) a trained BearingCNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"Baseline checkpoint to project (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--layers", nargs="+", default=list(DEFAULT_LAYERS),
        help=f"Conv1d layers to split (default: {' '.join(DEFAULT_LAYERS)})"
    )
    parser.add_argument(
        "--keep", type=float,
        help="Fraction of detail to keep, e.g. 0.5, 0.25, 0.1"
    )
    parser.add_argument(
        "--mode", choices=("rank", "energy"), default="rank",
        help="rank = fraction of each layer's rank, energy = fraction of singular-value energy"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Only print layer sizes and SVD spectra, then exit"
    )
    parser.add_argument(
        "--epochs", type=int, default=FINETUNE_EPOCHS,
        help=f"Fine-tuning epochs after projection (default: {FINETUNE_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=FINETUNE_LR,
        help=f"Fine-tuning learning rate (default: {FINETUNE_LR})"
    )
    parser.add_argument(
        "--no-finetune", action="store_true",
        help="Skip the brief retraining step (for before/after comparisons)"
    )
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Also score the test split (final reporting only, never tuning)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save the projected checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save projected checkpoints (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args(argv)

    if not args.report:
        if args.keep is None:
            parser.error("--keep is required unless --report is given")
        if not 0.0 < args.keep <= 1.0:
            parser.error("--keep must be in (0, 1]")

    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(SEED)

    device = torch.device("cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    print("Checkpoint:", args.checkpoint)
    print("Augmentation:", checkpoint.get("augmentation", "none"))
    print_layer_report(model)

    if args.report:
        print_spectrum_report(model, args.layers)
        return

    print(f"\nProjecting {', '.join(args.layers)} | mode={args.mode} keep={args.keep:g}")

    loaders = build_eval_loaders(checkpoint, include_test=args.eval_test)
    baseline = measure(model, loaders, device)

    projected, ranks, summary = project_model(model, args.layers, args.keep, args.mode)
    print_projection_summary(summary)

    val_loss, val_acc = evaluate(projected, loaders["val"], device)
    print(f"\nAfter projection (no fine-tune) | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    best_epoch = 0
    if not args.no_finetune:
        print(f"\nFine-tuning for {args.epochs} epochs at lr={args.lr}")
        best_epoch = finetune(projected, checkpoint, loaders["val"], args.epochs, args.lr, device)

    results = measure(projected, loaders, device)

    print("\n--- Results ---")
    print_results("Baseline", baseline)
    print_results("Projected", results)
    print(
        f"Parameters kept: {results['params'] / baseline['params']:.1%} | "
        f"Compression ratio (.pth): {baseline['size_mb'] / results['size_mb']:.2f}x"
    )

    if not args.no_save:
        out_path = save_projected_checkpoint(
            projected, checkpoint, args, ranks, summary, best_epoch, results
        )
        print("Projected checkpoint:", out_path)


if __name__ == "__main__":
    main()
