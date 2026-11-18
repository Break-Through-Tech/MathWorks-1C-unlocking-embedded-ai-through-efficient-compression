"""Reserve a unique, non-overwriting directory for a CNN training run."""

from hashlib import sha256
from pathlib import Path


def reserve_run_paths(output_dir, model, optimizer, *, seed, augmentation, batch_size, epochs):
    # Architecture and optimizer fingerprints distinguish configurations even
    # when their parameter counts or learning rates happen to match.
    arch_id = sha256(repr(model).encode("utf-8")).hexdigest()[:10]
    opt_id = sha256(repr(sorted(optimizer.defaults.items())).encode("utf-8")).hexdigest()[:10]
    lr = format(optimizer.param_groups[0]["lr"], ".8g").replace(".", "p")
    aug = augmentation.replace(".", "p")
    run_id = (
        f"cnn_{aug}_seed{seed}_{optimizer.__class__.__name__.lower()}"
        f"_lr{lr}_bs{batch_size}_ep{epochs}_m{arch_id}_o{opt_id}"
    )
    run_dir = Path(output_dir) / run_id

    # An exclusive mkdir is atomic: reruns/concurrent runs of this configuration
    # fail BEFORE training and cannot replace an existing run's outputs.
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run already reserved: {run_dir}. No training started. "
            "Use a new configuration or manually archive an abandoned run."
        ) from exc

    return run_dir / "checkpoint.pth", run_dir / "history.csv"
