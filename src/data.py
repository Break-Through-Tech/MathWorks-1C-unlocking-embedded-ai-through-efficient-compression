"""
src/data.py

Shared data-loading utilities for the bearing fault diagnosis project.

This module is the SINGLE source of truth for how data enters the pipeline.
Every teammate (classical models, deep models, compression experiments)
should load data through `load_split()` / `load_all_splits()` so that:

  1. Everyone uses the exact same train/val/test split (already fixed by
     the provided .mat files — we NEVER re-shuffle or re-split here).
  2. Everyone reads the same .mat keys, so no one silently loads mismatched
     data due to a typo or a different assumption about key names.

Expected .mat file structure (confirmed from data/train.mat, val.mat, test.mat):
    trainData   : (393, 5000) float array — 393 windows, 5000 samples each
    trainLabels : (393,)      string labels, e.g. 'Normal', 'InnerRaceFault', 'OuterRaceFault'
    valData     : (27, 5000)
    valLabels   : (27,)
    testData    : (102, 5000)
    testLabels  : (102,)

If your actual .mat files use different key names, update _KEY_MAP below —
that is the ONLY place key names should ever be hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat

Split = Literal["train", "val", "test"]

# ---------------------------------------------------------------------------
# Single place to define the exact .mat key names for each split.
# If your files differ (e.g. "trainX"/"trainY"), change ONLY this dict.
# ---------------------------------------------------------------------------
_KEY_MAP: dict[Split, dict[str, str]] = {
    "train": {"data": "trainData", "labels": "trainLabels"},
    "val":   {"data": "valData",   "labels": "valLabels"},
    "test":  {"data": "testData",  "labels": "testLabels"},
}

_DEFAULT_FILENAMES: dict[Split, str] = {
    "train": "train.mat",
    "val":   "val.mat",
    "test":  "test.mat",
}


def _clean_labels(raw_labels: np.ndarray) -> np.ndarray:
    """
    scipy.io.loadmat frequently returns MATLAB string/cell arrays as messy
    nested numpy arrays (e.g. dtype=object, or dtype='<U...' wrapped in an
    extra dimension). This normalizes them into a clean 1D numpy array of
    plain Python strings, e.g. array(['Normal', 'InnerRaceFault', ...]).
    """
    flat = np.asarray(raw_labels).squeeze()

    cleaned = []
    for item in flat:
        # Unwrap nested arrays/lists (common for MATLAB cell arrays of char)
        while isinstance(item, np.ndarray):
            item = item.item() if item.size == 1 else item[0]
        cleaned.append(str(item).strip())

    return np.array(cleaned)


def load_split(
    split: Split,
    data_dir: str | Path = "data",
    filename: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one split (train, val, or test) from its .mat file.

    Parameters
    ----------
    split : "train" | "val" | "test"
    data_dir : folder containing train.mat / val.mat / test.mat
    filename : override the default filename for this split if needed

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5000), dtype float
    y : np.ndarray, shape (n_samples,), dtype str
        Plain string class labels, e.g. 'Normal', 'InnerRaceFault', 'OuterRaceFault'.

    Notes
    -----
    This function does NOT shuffle, resplit, standardize, or augment the
    data. It only loads exactly what is in the file, under the given split.
    Standardization and augmentation are separate, explicit steps — see
    `standardize()` and `augment()` below — so that it's always clear
    what has and hasn't been applied to a given array.
    """
    if split not in _KEY_MAP:
        raise ValueError(f"split must be one of {list(_KEY_MAP)}, got {split!r}")

    data_dir = Path(data_dir)
    fname = filename or _DEFAULT_FILENAMES[split]
    path = data_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Expected {split}.mat under {data_dir}/ "
            f"(pass filename= if it's named differently)."
        )

    mat = loadmat(path)

    keys = _KEY_MAP[split]
    if keys["data"] not in mat:
        raise KeyError(
            f"Key '{keys['data']}' not found in {path}. "
            f"Available keys: {[k for k in mat if not k.startswith('__')]}. "
            f"Update _KEY_MAP in data.py if the real key name differs."
        )
    if keys["labels"] not in mat:
        raise KeyError(
            f"Key '{keys['labels']}' not found in {path}. "
            f"Available keys: {[k for k in mat if not k.startswith('__')]}. "
            f"Update _KEY_MAP in data.py if the real key name differs."
        )

    X = np.asarray(mat[keys["data"]], dtype=float)
    y = _clean_labels(mat[keys["labels"]])

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split}: X has {X.shape[0]} rows but y has {y.shape[0]} labels — "
            f"these should match. Check the .mat file / _KEY_MAP."
        )

    return X, y


def load_all_splits(
    data_dir: str | Path = "data",
) -> dict[Split, tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper: load train, val, and test in one call.

    Returns
    -------
    dict like {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    """
    return {split: load_split(split, data_dir=data_dir) for split in ("train", "val", "test")}


def standardize(
    X_train: np.ndarray,
    *arrays: np.ndarray,
    method: Literal["zscore_global", "zscore_per_sample"] = "zscore_global",
) -> tuple[np.ndarray, ...]:
    """
    Standardize signal data before model training.

    *** PLACEHOLDER — to be finalized with Person 5's conclusion. ***
    The dataset docs confirm standardization is the one required
    preprocessing step not already done in the .mat files. The exact
    method (global z-score vs. per-sample z-score vs. something else,
    and whether it's per-channel/per-feature) should be set based on
    Person 5's analysis. Until then, this defaults to a standard,
    reasonable choice: global z-score using train-set statistics only.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 5000)
        The array whose mean/std define the standardization — MUST be the
        train set. Val/test are transformed using train's statistics, never
        their own, to avoid leaking information from val/test into training.
    *arrays : additional np.ndarray of shape (n, 5000), e.g. X_val, X_test.
        Transformed using X_train's statistics.
    method : "zscore_global" | "zscore_per_sample"
        - "zscore_global": subtract/divide by ONE mean and ONE std computed
          across all of X_train (all samples, all timesteps). Simple,
          assumes signal scale is consistent across the dataset.
        - "zscore_per_sample": each individual signal (row) is standardized
          using its own mean/std. Useful if different recordings have very
          different absolute amplitude/offset, but discards absolute
          amplitude as a signal — discuss with Person 5 before using this.

    Returns
    -------
    Tuple of standardized arrays, same order as input:
        (X_train_std,) if no extra arrays given, else (X_train_std, arr1_std, arr2_std, ...)

    Example
    -------
        X_train_std, X_val_std, X_test_std = standardize(X_train, X_val, X_test)
    """
    all_arrays = (X_train,) + arrays

    if method == "zscore_global":
        mean = X_train.mean()
        std = X_train.std()
        std = std if std > 1e-12 else 1.0  # guard against divide-by-zero
        return tuple((arr - mean) / std for arr in all_arrays)

    elif method == "zscore_per_sample":
        def _per_sample(arr: np.ndarray) -> np.ndarray:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            std = np.where(std > 1e-12, std, 1.0)
            return (arr - mean) / std
        return tuple(_per_sample(arr) for arr in all_arrays)

    else:
        raise ValueError(f"Unknown method: {method!r}")


def augment(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to training data.

    *** PLACEHOLDER — to be filled in with Person 2's findings. ***
    This hook exists so the pipeline has one obvious, agreed-upon place to
    plug in augmentation once Person 2 determines what works for vibration
    signal data (e.g. jitter/noise injection, time-shift, scaling, etc.).

    IMPORTANT: only ever call this on the TRAINING split. Never augment
    val or test data — that would corrupt evaluation.

    Parameters
    ----------
    X, y : the training data/labels to augment.
    seed : int
        Fixes the random state so augmentation is reproducible across runs
        and across teammates. Always pass an explicit seed when calling
        this — never rely on global numpy random state.

    Returns
    -------
    (X_augmented, y_augmented) — currently a no-op passthrough returning
    X, y unchanged. Replace the body once Person 2's method is decided.
    """
    rng = np.random.default_rng(seed)  # noqa: F841 -- reserved for future use

    # --- No-op until Person 2's augmentation method is plugged in here ---
    X_augmented, y_augmented = X, y

    return X_augmented, y_augmented


if __name__ == "__main__":
    # Quick manual sanity check — mirrors the environment check output
    # you already saw (shapes + class balance).
    from collections import Counter

    splits = load_all_splits()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("Counts:", len(y_train), len(y_val), len(y_test))
    print("Train class balance:", Counter(y_train))
    print("Val class balance:", Counter(y_val))
    print("Test class balance:", Counter(y_test))
