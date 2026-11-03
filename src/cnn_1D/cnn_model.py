
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

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 input channel -> 16 learned feature channels
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 2: 16 -> 32 channels
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 3: 32 -> 64 channels
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 4: 64 -> 96 channels
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.20),
            nn.Linear(96, 3),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
