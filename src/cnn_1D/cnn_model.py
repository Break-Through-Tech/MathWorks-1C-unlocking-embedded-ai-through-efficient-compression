
import torch
from torch import nn


# Fixed class order for training, evaluation, and checkpoint loading.
CLASS_NAMES = ("Normal", "InnerRaceFault", "OuterRaceFault")


class BearingCNN(nn.Module):
    """
    A compact 1D CNN for classifying bearing vibration signals.

    Input:  (batch_size, 1, 5000)
    Output: (batch_size, 3) raw classification logits
    """

    def __init__(self, channels=(16, 32, 64, 96), dropout=0.20):
        super().__init__()

        if len(channels) != 4 or any(c <= 0 for c in channels):
            raise ValueError("channels must contain four positive integers")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        c1, c2, c3, c4 = channels

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, c1, kernel_size=15, padding=7),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 2
            nn.Conv1d(c1, c2, kernel_size=9, padding=4),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 3
            nn.Conv1d(c2, c3, kernel_size=7, padding=3),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 4
            nn.Conv1d(c3, c4, kernel_size=5, padding=2),
            nn.BatchNorm1d(c4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c4, 3),
        )


    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
