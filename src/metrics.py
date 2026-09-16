"""
src/metrics.py

Shared evaluation metrics for the bearing fault diagnosis project.

This module is the SINGLE source of truth for how the team measures model
quality. Every teammate (classical models, deep models, compressed
variants) should evaluate through these functions so that results are
directly comparable across the whole team:

    - accuracy()                   : % correct predictions
    - measure_model_size_mb()      : model size on disk, in MB
    - measure_inference_latency_ms(): ms per sample, batch size = 1

Design goal: all three functions work the same way regardless of whether
the model is "classical" (e.g. sklearn SVM / Random Forest) or "deep"
(PyTorch / TensorFlow-Keras). This matters because in October the team
will be comparing a baseline model against pruned/quantized/projected
versions of it, possibly across different frameworks — the comparison is
only fair if size and speed are measured identically every time.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

ModelKind = Literal["sklearn", "pytorch", "keras", "path"]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of correct predictions, in [0.0, 1.0].

    Works for any label type (strings like 'Normal'/'InnerRaceFault', or
    integer class indices) as long as y_true and y_pred use the same
    label encoding.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,) — ground-truth labels
    y_pred : array-like, shape (n_samples,) — predicted labels

    Returns
    -------
    float accuracy, e.g. 0.9412
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} samples but y_pred has "
            f"{y_pred.shape[0]} — they must match."
        )
    if y_true.shape[0] == 0:
        raise ValueError("y_true/y_pred are empty — cannot compute accuracy.")

    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# Model size on disk
# ---------------------------------------------------------------------------

def _dir_size_bytes(path: Path) -> int:
    """Sum of file sizes under path (handles both a single file and a folder,
    e.g. a TensorFlow SavedModel directory, which is multiple files)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if fp.exists():
                total += fp.stat().st_size
    return total


def measure_model_size_mb(
    model: Any = None,
    path: str | Path | None = None,
    kind: ModelKind = "path",
) -> float:
    """
    Measure a model's size on disk, in MB. Works for classical (sklearn)
    and deep (PyTorch, Keras/TensorFlow) models, or any already-saved
    file/folder.

    This is the metric that matters for embedded deployment: what actually
    matters is how many bytes the model occupies on the target device's
    flash storage — not any in-memory Python object size.

    Two ways to call this:

    (A) You already have the model saved somewhere (e.g. a compressed/
        quantized model, or a TFLite file, or a SavedModel folder):
            measure_model_size_mb(path="models/pruned_model.pth")
            measure_model_size_mb(path="models/saved_model_dir")  # folder ok

    (B) You have a live model object and want this function to save it
        (to a temp location) and measure it for you:
            measure_model_size_mb(model=my_sklearn_model, kind="sklearn")
            measure_model_size_mb(model=my_torch_model, kind="pytorch")
            measure_model_size_mb(model=my_keras_model, kind="keras")

    Parameters
    ----------
    model : a trained model object (required if kind != "path")
    path  : path to an already-saved model file or folder
            (required if kind == "path", the default)
    kind  : "path" | "sklearn" | "pytorch" | "keras"
            Tells this function how to save `model` before measuring.
            Ignored (and unnecessary) if you pass `path` directly.

    Returns
    -------
    float — size in MB (decimal MB, i.e. bytes / 1_000_000, which is the
    convention used for storage/flash capacity specs; NOT MiB / 1024^2).

    Notes
    -----
    When `model` is given (kind != "path"), the model is saved to a
    temporary file/folder purely to measure its size, then deleted — this
    function never leaves stray files behind and never overwrites your
    real saved models.
    """
    if kind == "path":
        if path is None:
            raise ValueError('kind="path" requires `path=` to an existing file or folder.')
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No such file or folder: {p}")
        size_bytes = _dir_size_bytes(p)
        return size_bytes / 1_000_000

    if model is None:
        raise ValueError(f'kind="{kind}" requires `model=` (the trained model object).')

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if kind == "sklearn":
            fpath = tmp_path / "model.pkl"
            with open(fpath, "wb") as f:
                pickle.dump(model, f)
            size_bytes = _dir_size_bytes(fpath)

        elif kind == "pytorch":
            import torch  # local import: only required if this path is used
            fpath = tmp_path / "model.pth"
            torch.save(model.state_dict(), fpath)
            size_bytes = _dir_size_bytes(fpath)

        elif kind == "keras":
            # Keras SavedModel format is a directory of files — measure the
            # whole directory, since that's what would actually ship.
            save_dir = tmp_path / "saved_model"
            model.save(save_dir)
            size_bytes = _dir_size_bytes(save_dir)

        else:
            raise ValueError(
                f"Unknown kind={kind!r}. Use one of: 'path', 'sklearn', 'pytorch', 'keras'."
            )

        return size_bytes / 1_000_000


# ---------------------------------------------------------------------------
# Inference speed: ms per sample @ batch size = 1
# ---------------------------------------------------------------------------

def measure_inference_latency_ms(
    predict_fn: Callable[[np.ndarray], Any],
    X: np.ndarray,
    n_warmup: int = 10,
    n_repeats: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """
    Measure inference speed at batch size = 1 (i.e. one sample at a time —
    the realistic scenario for an embedded device reacting to a single
    incoming sensor reading, as opposed to a large batched throughput test).

    Parameters
    ----------
    predict_fn : callable that takes ONE sample shaped (1, n_features)
        (or whatever shape your model expects for a single sample) and
        returns a prediction. This is deliberately framework-agnostic:
            sklearn:  predict_fn = lambda x: sk_model.predict(x)
            pytorch:  predict_fn = lambda x: torch_model(torch.tensor(x, dtype=torch.float32))
            keras:    predict_fn = lambda x: keras_model.predict(x, verbose=0)
        Wrap your model however is needed so predict_fn(sample) just works.
    X : np.ndarray, shape (n_samples, n_features)
        Pool of samples to draw single-sample inputs from for timing.
    n_warmup : int
        Number of untimed calls run first, to let caches/JIT/lazy-init
        settle before timing starts. Warmup runs are excluded from the
        reported statistics.
    n_repeats : int
        Number of timed single-sample calls to average over.
    seed : int
        Controls which samples get drawn from X, for reproducibility.

    Returns
    -------
    dict with:
        "mean_ms"   : average ms per sample
        "std_ms"    : standard deviation across repeats
        "min_ms"    : fastest single-sample call
        "max_ms"    : slowest single-sample call
        "n_repeats" : how many timed calls this is based on

    Notes
    -----
    Timing is done with time.perf_counter(), the standard high-resolution
    timer for benchmarking in Python. Each call is timed individually
    (not "total time / n") so we also get variance, not just a mean —
    useful since embedded systems often care about worst-case latency too.
    """
    if X.shape[0] == 0:
        raise ValueError("X is empty — need at least one sample to time.")

    rng = np.random.default_rng(seed)
    n_available = X.shape[0]

    # Sample indices (with replacement if n_repeats/n_warmup exceed n_available)
    idx_warmup = rng.integers(0, n_available, size=n_warmup)
    idx_timed = rng.integers(0, n_available, size=n_repeats)

    # Warmup — not timed
    for i in idx_warmup:
        sample = X[i : i + 1]  # keep shape (1, n_features), i.e. batch of 1
        predict_fn(sample)

    # Timed runs
    times_ms = np.empty(n_repeats, dtype=float)
    for j, i in enumerate(idx_timed):
        sample = X[i : i + 1]
        start = time.perf_counter()
        predict_fn(sample)
        end = time.perf_counter()
        times_ms[j] = (end - start) * 1000.0

    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
        "n_repeats": n_repeats,
    }


if __name__ == "__main__":
    # Minimal smoke test using pure numpy so this runs with zero extra
    # dependencies (no sklearn/torch/keras required just to sanity-check
    # the module imports and functions behave as expected).

    # --- accuracy() ---
    y_true = np.array(["Normal", "InnerRaceFault", "OuterRaceFault", "Normal"])
    y_pred = np.array(["Normal", "InnerRaceFault", "Normal", "Normal"])
    acc = accuracy(y_true, y_pred)
    print(f"accuracy() smoke test: {acc:.4f} (expected 0.75)")

    # --- measure_model_size_mb() via a plain saved file ---
    with tempfile.TemporaryDirectory() as tmp:
        fake_model_path = Path(tmp) / "fake_model.bin"
        fake_model_path.write_bytes(b"0" * 2_000_000)  # exactly 2,000,000 bytes
        size_mb = measure_model_size_mb(path=fake_model_path)
        print(f"measure_model_size_mb() smoke test: {size_mb:.4f} MB (expected 2.0000)")

    # --- measure_inference_latency_ms() with a trivial dummy predictor ---
    X_dummy = np.random.randn(50, 5000)

    def dummy_predict(x: np.ndarray):
        return x.sum()  # cheap, deterministic-ish stand-in for a real model

    stats = measure_inference_latency_ms(dummy_predict, X_dummy, n_warmup=5, n_repeats=20)
    print(f"measure_inference_latency_ms() smoke test: {stats}")
