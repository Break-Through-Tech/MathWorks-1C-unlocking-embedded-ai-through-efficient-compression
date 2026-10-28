# 2D Spectrogram CNN — Baseline & Comparison

This folder contains the 2D Spectrogram CNN implementation, training pipeline, and baseline comparison against the 1D CNN and SVM baseline.

## 1. Files

| File | Purpose |
|---|---|
| `cnn_2d_model.py` | STFT spectrogram conversion (`129x22` log-dB) and `TinySpecCNN` architecture |
| `cnn_2d_train.py` | Standardized spectrogram training loop and checkpointing |
| `cnn_2d_evaluate.py` | End-to-end evaluation (accuracy, size, latency) and logging to `results/experiments.csv` |

Saved model weights (`cnn_2d_spec.pth`) and training history (`cnn_2d_spec_history.csv`) are stored in `cnn_outputs/`.

## 2. Test Set Comparison

| Model | Test Accuracy | Size (MB) | Latency (ms/sample) |
|---|---:|---:|---:|
| **SVM (Baseline)** | 98.04% (`0.980392`) | `0.004960` | `0.175298` |
| **1D CNN (Raw Signal)** | 100.00% (`1.000000`) | `0.213313` | `2.531301` |
| **2D CNN (Spectrogram)** | 98.04% (`0.980392`) | `0.038525` | `2.257220` |

*Note: 2D CNN inference latency includes the per-sample STFT spectrogram computation overhead.*

## 3. Recommendation: Raw Signal vs. Spectrogram
Based on our baseline comparison, I recommend prioritizing the raw 1D signal (1D CNN) as our primary target for model compression, while keeping the 2D Spectrogram CNN as an ultra-compact benchmark. Although the 2D Spectrogram CNN (`TinySpecCNN`) is 5.5× smaller (`0.0385 MB` vs. `0.2133 MB`) and faster even with spectrogram computation included (`2.26 ms/sample` vs. `2.53 ms/sample`), the 1D CNN achieves 100% test accuracy compared to 98.04% for both the 2D CNN and SVM baseline. Furthermore, ingesting standardized raw 1D windows avoids the fixed computational overhead of an on-device FFT/spectrogram pipeline, and its larger parameter count (`50,819` parameters, `0.2133 MB`) gives us substantial room to apply pruning and quantization with the clear goal of compressing the 1D CNN down to the 2D CNN's `~0.038 MB` footprint without losing its 100% accuracy.
