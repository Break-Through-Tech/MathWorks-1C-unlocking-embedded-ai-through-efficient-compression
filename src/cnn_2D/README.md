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
| **1D CNN (Raw Signal)** | 100.00% (`1.000000`) | `0.213313` | `1.352639` |
| **2D CNN (Spectrogram)** | 100.00% (`1.000000`) | `0.038525` | `1.082752` |

*Note: 2D CNN inference latency includes the per-sample STFT spectrogram computation overhead.*

## 3. Recommendation: Raw Signal vs. Spectrogram
Both the 1D CNN and the 2D Spectrogram CNN achieve a perfect 100% test accuracy, successfully outperforming the SVM baseline. The 2D CNN (`TinySpecCNN`) is highly efficient, operating at a footprint 5.5× smaller (`0.0385 MB` vs. `0.2133 MB`) and slightly faster (`1.08 ms/sample` vs. `1.35 ms/sample`) even when STFT computation time is included. However, because neural network compression cannot shrink the fixed pre-processing math of an STFT, the raw 1D CNN remains our primary target for pruning and quantization. Our goal for the compression pipeline will be to shrink the 1D CNN below the 2D CNN's `0.038 MB` footprint while avoiding the fixed computational overhead of spectrogram generation.


## Part 2: Noise Robustness Stress Test

![2D CNN Noise Stress Test](../cnn_outputs/noise_stress_test_chart.png)

Across a 10-trial Monte Carlo noise evaluation, the 2D Spectrogram CNN (`TinySpecCNN`) demonstrates exceptional resilience to background interference, maintaining 99.5%+ mean test accuracy from a clean baseline down to 5 dB SNR. This robustness occurs because the Short-Time Fourier Transform (STFT) distributes random broadband Gaussian noise evenly across all 129 frequency bins, allowing the localized bearing-fault harmonics to remain clearly distinguishable after log-dB scaling. Slight degradation only begins at extreme noise levels (0 dB SNR), where the static noise power fully equals the raw signal power, pushing the accuracy down slightly to 97.0%.
