import numpy as np
import torch
import torch.nn as nn
from scipy.signal import spectrogram

# transform 1D arrays into 2D spectrogram images
FS = 48828 # the sampling rate which is how many vibration readings the sensor took per second

def signals_to_spectrograms(X, fs=FS, nperseg=256):
    specs = []
    for sig in X:
        # scipy's spectrogram breaks your 5000-point 1D signal into small overlapping time windows.
        # f = frequency bins (this becomes the 129-pixel "height" of your image)
        # t = time segments (this becomes the 22-pixel "width" of your image)
        # Sxx = the raw intensity of the frequencies at that time step
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg)

        # Convert the raw intensity into decibels (dB)
        # This is like adjusting the contrast on an image—it makes the quiet
        # background noise fade out and the loud fault frequencies pop visually.
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        specs.append(Sxx_db)

    # np.expand_dims adds a "color channel" to your data array.
    # Convolutional Neural Networks (CNNs) expect images to have color channels (like RGB has 3).
    # Because our spectrograms are basically heatmaps, we add an axis of '1' to act as "grayscale".
    return np.expand_dims(np.array(specs), axis=1)


class TinySpecCNN(nn.Module):
    """
    A lightweight 2D CNN designed for embedded deployment.
    Keeps filter counts low (8, 16) to minimize the uncompressed .pth size (MB)
    and reduce inference latency (ms/sample) prior to quantization.
    """
    def __init__(self):
        super().__init__()

        # Block 1: Input (Batch, 1, 129, 22)
        # padding=1 preserves spatial dimensions during the convolution.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # MaxPool2d(2) halves the spatial dimensions:
        # Height: 129 // 2 = 64. Width: 22 // 2 = 11.
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Output shape after Block 1: (Batch, 8, 64, 11)

        # Block 2: Input (Batch, 8, 64, 11)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # MaxPool2d(2) halves the spatial dimensions again:
        # Height: 64 // 2 = 32. Width: 11 // 2 = 5.
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Output shape after Block 2: (Batch, 16, 32, 5)

        # Fully Connected Classifier
        # Flattened spatial features: 16 (channels) * 32 (height) * 5 (width) = 2560
        self.fc = nn.Linear(in_features=2560, out_features=3)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten from (Batch, 16, 32, 5) to (Batch, 2560)
        x = x.view(x.size(0), -1)

        # Classification (Logits, since CrossEntropyLoss applies Softmax internally)
        x = self.fc(x)
        return x
