
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import _time_shift, _add_noise
from src.cnn_1D.cnn_model import CLASS_NAMES


class AugmentedTrainingDataset(Dataset):
    """
    Training-only augmentation for bearing vibration windows.

    Each example independently has:
      - 50% chance of circular shift within +/-500 samples
      - 50% chance of Gaussian noise at 20 dB SNR

    Transformations happen on raw signals, followed by
    fixed clean-training global standardization.
    """

    def __init__(self, X, y, mean, std, seed=42):
        self.X = np.asarray(X, dtype=np.float64)
        self.mean = float(mean)
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

        label_map = {
            name: i for i, name in enumerate(CLASS_NAMES)
        }

        self.y = np.array(
            [label_map[str(label)] for label in y],
            dtype=np.int64
        )

        assert self.X.shape == (393, 5000)
        assert len(self.y) == len(self.X)
        assert self.std > 0
        assert np.isfinite(self.X).all()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        signal = self.X[index].copy()

        # These helpers come from the team's shared data.py.
        if self.rng.random() < 0.5:
            signal = _time_shift(
                signal, self.rng, max_shift=500
            )

        if self.rng.random() < 0.5:
            signal = _add_noise(
                signal, self.rng, snr_db=20.0
            )

        # Always use original clean-training statistics.
        signal = np.ascontiguousarray(
            ((signal - self.mean) / self.std)[None, :],
            dtype=np.float32
        )

        target = int(self.y[index])

        return (
            torch.from_numpy(signal),
            torch.tensor(target, dtype=torch.long)
        )
