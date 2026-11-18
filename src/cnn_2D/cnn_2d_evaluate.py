import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from src.data import load_all_splits
from src.metrics import measure_model_size_mb, measure_inference_latency_ms, accuracy
from src.cnn_2D.cnn_2d_model import TinySpecCNN, signals_to_spectrograms, FS

# Data Loading & Standardization
# Load the raw train and test splits to prepare our evaluation data
splits = load_all_splits(data_dir='data')
X_train, y_train = splits["train"]
X_test, y_test = splits["test"]

# Map string labels to integer indices for accuracy scoring
label_map = {'Normal': 0, 'InnerRaceFault': 1, 'OuterRaceFault': 2}
y_test_idx = np.array([label_map[y] for y in y_test])

# Standardize the test set using ONLY the training set's mean and standard deviation
train_mean = X_train.mean()
train_std = X_train.std()
X_test_std = (X_test - train_mean) / train_std


# Model Architectures
class Conv1DModel(nn.Module):
    """The 1D CNN architecture used in previous baseline experiments."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Identity(), nn.Identity(), nn.Linear(96, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Load Saved Weights
# Initialize both models and load their respective trained state dictionaries.
# .eval() freezes dropout and batch normalization layers for deterministic testing.
model_1d = Conv1DModel()
model_1d.load_state_dict(torch.load('cnn_outputs/cnn_rough_seed42.pth', map_location='cpu')['model_state_dict'], strict=True)
model_1d.eval()

model_2d = TinySpecCNN()
model_2d.load_state_dict(torch.load('cnn_outputs/cnn_2d_spec.pth', map_location='cpu'))
model_2d.eval()


# Evaluate 1D CNN
# Format 1D input shape to (Batch, Channels, Features) -> (102, 1, 5000)
X_test_1d = np.expand_dims(X_test_std, axis=1)

with torch.no_grad():
    outputs_1d = model_1d(torch.tensor(X_test_1d, dtype=torch.float32))
    preds_1d = torch.max(outputs_1d, 1)[1] # Get index of max logit

acc_1d = accuracy(y_test_idx, preds_1d.numpy())
size_1d = measure_model_size_mb(model_1d, kind="pytorch")

def predict_1d(sample):
    """Wrapper function to measure 1D CNN latency for a single sample."""
    with torch.no_grad():
        return model_1d(torch.tensor(np.expand_dims(sample, axis=0), dtype=torch.float32))

lat_1d = measure_inference_latency_ms(predict_1d, X_test_std)['mean_ms']


# Evaluate 2D CNN (Including Spectrogram Overhead)
# Generate test spectrograms and score accuracy
X_test_spec = signals_to_spectrograms(X_test_std)
with torch.no_grad():
    outputs_2d = model_2d(torch.tensor(X_test_spec, dtype=torch.float32))
    preds_2d = torch.max(outputs_2d, 1)[1]

acc_2d = accuracy(y_test_idx, preds_2d.numpy())
size_2d = measure_model_size_mb(model_2d, kind="pytorch")

def predict_2d(sample):
    """
    Wrapper function to measure 2D CNN latency for a single sample.
    Crucially, this INCLUDES the time it takes to compute the spectrogram,
    ensuring a fair comparison against models that ingest raw 1D data.
    """
    spec = 10 * np.log10(spectrogram(sample[0], fs=FS, nperseg=256)[2] + 1e-10)
    spec_input = np.expand_dims(np.expand_dims(spec, axis=0), axis=0)
    with torch.no_grad():
        return model_2d(torch.tensor(spec_input, dtype=torch.float32))

lat_2d = measure_inference_latency_ms(predict_2d, X_test_std)['mean_ms']


# Compile & Display Results
# Read the team's master CSV to pull the latest SVM baseline metrics
csv_path = 'results/experiments.csv'
exp_df = pd.read_csv(csv_path)
svm_row = exp_df[exp_df['model'].str.contains('SVM', na=False)].iloc[-1]

comparison_df = pd.DataFrame({
    'Model': ['SVM (Baseline)', '1D CNN', '2D CNN (Spectrogram)'],
    'Test Accuracy': [svm_row['accuracy_test'], acc_1d, acc_2d],
    'Size (MB)': [svm_row['model_size_mb'], size_1d, size_2d],
    'Latency (ms/sample)': [svm_row['latency_ms_per_sample'], lat_1d, lat_2d]
})

print(comparison_df.to_string(index=False))
