"""Generate fixed, validation-only bearing-signal perturbations."""

from hashlib import sha256
from pathlib import Path

import numpy as np

from src.data import load_split, _add_noise


BASE_SEED = 20261107
SNR_LEVELS = (20, 15, 10, 5, 0)
SHIFT_OFFSETS = (-500, -250, 250, 500)
REALIZATIONS = 3

OUTPUT_PATH = Path(
    "cnn_outputs/robustness/validation_perturbations_v1.npz"
)


def main():
    X, y = load_split("val")
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=str)

    assert X.shape == (27, 5000)
    assert y.shape == (27,)
    assert np.isfinite(X).all()

    fingerprint = sha256(X.tobytes()).hexdigest()

    arrays = {
        "X_clean": X,
        "y": y,
        "base_seed": np.array(BASE_SEED),
        "validation_sha256": np.array(fingerprint),
    }

    # Fixed circular shifts on raw signals.
    for offset in SHIFT_OFFSETS:
        direction = "m" if offset < 0 else "p"
        key = f"X_shift_{direction}{abs(offset)}"
        arrays[key] = np.roll(X, offset, axis=1)

    # Independent deterministic noise realizations.
    for level_index, snr in enumerate(SNR_LEVELS):
        for realization in range(REALIZATIONS):
            seed = BASE_SEED + 100 * level_index + realization
            rng = np.random.default_rng(seed)

            key = f"X_noise_{snr}db_r{realization}"

            arrays[key] = np.stack([
                _add_noise(signal, rng, snr_db=float(snr))
                for signal in X
            ])

    conditions = sorted(
        key for key in arrays if key.startswith("X_")
    )
    assert len(conditions) == 20

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing benchmark.
    with OUTPUT_PATH.open("xb") as file:
        np.savez_compressed(file, **arrays)

    # Verify the saved dataset.
    with np.load(OUTPUT_PATH, allow_pickle=False) as saved:
        assert len([
            key for key in saved.files if key.startswith("X_")
        ]) == 20

        assert np.array_equal(saved["X_clean"], X)
        assert np.array_equal(saved["y"], y)

        for offset in SHIFT_OFFSETS:
            direction = "m" if offset < 0 else "p"
            key = f"X_shift_{direction}{abs(offset)}"
            assert np.array_equal(
                saved[key], np.roll(X, offset, axis=1)
            )

        for key in conditions:
            assert saved[key].shape == (27, 5000)
            assert np.isfinite(saved[key]).all()

        assert saved["validation_sha256"].item() == fingerprint

    print("VALIDATION ROBUSTNESS DATASET CREATED")
    print("Conditions:", len(conditions))
    print("Evaluations per model:", len(conditions) * len(X))
    print("Base seed:", BASE_SEED)
    print("Validation SHA256:", fingerprint)
    print("Output:", OUTPUT_PATH)
    print("ALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    main()
