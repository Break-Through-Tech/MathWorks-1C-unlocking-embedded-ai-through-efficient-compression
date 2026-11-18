"""
src/cnn_1D/projection_robustness.py

Projection Part 2, step 2: how well do the projected models hold up under
noise and time shifts, and does training with augmentation help?

Runs Gurbaj's validation robustness suite (docs/cnn_robustness_protocol.md
on feature/1d-cnn-optimization) on the two dense source models and every
projected package in cnn_outputs/projected/. The 20 perturbed versions of
the 27 validation windows are regenerated here with the same seeds and
the same shared noise helper as cnn_robustness_data.py, then checked
against the dense-model results Gurbaj recorded, so these numbers line up
with his. Validation only: the test split is never touched.

Writes results/projection_robustness.csv.

Usage (from the repository root):

    python -m src.cnn_1D.projection_robustness
"""

import csv
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import _add_noise, load_split
from src.cnn_1D.cnn_model import CLASS_NAMES
from src.cnn_1D.cnn_project import DEFAULT_OUTPUT_DIR, PACKAGE_NAME, load_model
from src.cnn_1D.projection_build import SOURCES

OUTPUT_CSV = Path("results/projection_robustness.csv")

# Same constants as Gurbaj's cnn_robustness_data.py.
BASE_SEED = 20261107
SNR_LEVELS = (20, 15, 10, 5, 0)
SHIFT_OFFSETS = (-500, -250, 250, 500)
REALIZATIONS = 3
VALIDATION_SHA256 = "0f2c523f2eb27c2189cd1654a5efbdccabf36851243ec7788ee6a697f0ed1d6c"

# Gurbaj's recorded results for the two dense sources
# (cnn_outputs/robustness/augmentation_robustness_v1.csv, seed 42):
# every condition is 27/27 except these.
REFERENCE_MISSES = {"X_noise_0db_r0": 18, "X_noise_0db_r1": 18,
                    "X_noise_0db_r2": 18, "X_noise_5db_r2": 26}
REFERENCE_LOSS_0DB_R0 = {"none": 1.5816679, "shift_p0.5_noise_p0.5_20dB": 1.5431443}

RUN_DIR_RE = re.compile(r"rank_(\d+(?:\.\d+)?)pct")


def build_conditions():
    """The 20 fixed perturbations of the raw validation windows."""
    X, y = load_split("val")
    X = np.ascontiguousarray(X, dtype=np.float64)

    if sha256(X.tobytes()).hexdigest() != VALIDATION_SHA256:
        raise ValueError("Validation split differs from the one Gurbaj's suite used")

    conditions = {"X_clean": X}
    for offset in SHIFT_OFFSETS:
        direction = "m" if offset < 0 else "p"
        conditions[f"X_shift_{direction}{abs(offset)}"] = np.roll(X, offset, axis=1)

    for level_index, snr in enumerate(SNR_LEVELS):
        for realization in range(REALIZATIONS):
            rng = np.random.default_rng(BASE_SEED + 100 * level_index + realization)
            conditions[f"X_noise_{snr}db_r{realization}"] = np.stack([
                _add_noise(signal, rng, snr_db=float(snr)) for signal in X
            ])

    assert len(conditions) == 20
    return conditions, np.asarray(y, dtype=str)


def describe(condition):
    """(kind, snr_db, realization, shift) for a condition name."""
    noise = re.fullmatch(r"X_noise_(\d+)db_r(\d)", condition)
    if noise:
        return "noise", int(noise.group(1)), int(noise.group(2)), ""
    shift = re.fullmatch(r"X_shift_([mp])(\d+)", condition)
    if shift:
        sign = -1 if shift.group(1) == "m" else 1
        return "shift", "", "", sign * int(shift.group(2))
    return "clean", "", "", ""


def models_to_evaluate():
    """(source, keep, path) for each dense source and its projected packages."""
    models = []
    for source in SOURCES:
        models.append((source, 1.0, source))
        source_dir = DEFAULT_OUTPUT_DIR / source.stem
        runs = []
        for run_dir in source_dir.glob("rank_*pct"):
            match = RUN_DIR_RE.fullmatch(run_dir.name)
            if match and (run_dir / PACKAGE_NAME).exists():
                runs.append((float(match.group(1)) / 100, run_dir / PACKAGE_NAME))
        models.extend((source, keep, path) for keep, path in sorted(runs, reverse=True))
    return models


def evaluate_conditions(model, checkpoint, conditions, targets):
    mean = float(checkpoint["train_mean"])
    std = float(checkpoint["train_std"])
    results = {}
    for name, X in conditions.items():
        inputs = torch.from_numpy(
            np.ascontiguousarray(((X - mean) / std)[:, None, :], dtype=np.float32)
        )
        with torch.inference_mode():
            logits = model(inputs)
        results[name] = (
            int((logits.argmax(dim=1) == targets).sum()),
            float(F.cross_entropy(logits, targets)),
        )
    return results


def check_against_reference(augmentation, results):
    """The dense sources must reproduce Gurbaj's recorded numbers."""
    for name, (correct, _) in results.items():
        expected = REFERENCE_MISSES.get(name, 27)
        assert correct == expected, (augmentation, name, correct, expected)
    # Gurbaj's losses were computed on GPU; CPU results differ by ~0.1%.
    loss = results["X_noise_0db_r0"][1]
    expected_loss = REFERENCE_LOSS_0DB_R0[augmentation]
    assert abs(loss - expected_loss) / expected_loss < 1e-2, (augmentation, loss, expected_loss)


def main():
    torch.set_num_threads(1)
    conditions, y = build_conditions()
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    targets = torch.tensor([label_map[label] for label in y], dtype=torch.long)

    rows = []
    for source, keep, path in models_to_evaluate():
        model, checkpoint = load_model(path)
        augmentation = checkpoint.get("augmentation", "none")
        results = evaluate_conditions(model, checkpoint, conditions, targets)

        if keep == 1.0:
            check_against_reference(augmentation, results)

        params = sum(p.numel() for p in model.parameters())
        for name, (correct, loss) in sorted(results.items()):
            kind, snr, realization, shift = describe(name)
            rows.append({
                "model": "augmented" if augmentation != "none" else "clean",
                "source": source.name,
                "keep": keep,
                "params": params,
                "artifact": path.as_posix(),
                "condition": name,
                "kind": kind,
                "snr_db": snr,
                "realization": realization,
                "shift": shift,
                "samples": len(y),
                "correct": correct,
                "accuracy": correct / len(y),
                "cross_entropy": loss,
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Dense sources reproduce Gurbaj's recorded robustness results: PASSED")
    print("\nMean accuracy over 3 noise realizations (27 validation windows each)")
    header = f"{'model':<10} {'keep':>5} {'params':>7} {'clean':>6} " + " ".join(
        f"{snr:>2}dB " for snr in SNR_LEVELS) + "  shifts"
    print(header)
    for source, keep, path in models_to_evaluate():
        sub = [r for r in rows if r["artifact"] == path.as_posix()]
        clean = next(r["accuracy"] for r in sub if r["kind"] == "clean")
        noise = [np.mean([r["accuracy"] for r in sub if r["snr_db"] == snr]) for snr in SNR_LEVELS]
        shifts = min(r["accuracy"] for r in sub if r["kind"] == "shift")
        print(f"{sub[0]['model']:<10} {keep:>5.2f} {sub[0]['params']:>7} {clean:>6.0%} "
              + " ".join(f"{a:>5.0%}" for a in noise) + f"  {shifts:>5.0%}")

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
