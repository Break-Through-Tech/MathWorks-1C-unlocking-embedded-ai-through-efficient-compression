# 1D CNN Bearing-Fault Classification — Rough Baseline

This folder contains the rough version of the 1D CNN developed in PyTorch. Both clean and augmented models have been trained and verified. Hyperparameter tuning, final evaluation, and compression testing still pending.

## 1. Model Handoff

Two preliminary checkpoints are available in `cnn_outputs/`:

| Checkpoint | Description |
|---|---|
| `cnn_rough_seed42.pth` | Model trained without augmentation |
| `cnn_rough_aug_seed42.pth` | Model trained with time shifts and Gaussian noise |

Neither has been formally selected as the reference for compression comparisons. The team should agree on which checkpoint to use before making those comparisons.

Both checkpoints contain the trained weights, normalization statistics, class mapping, training configuration, and selected epoch.

### Environment and Verification

Developed using Python 3.13, PyTorch 2.10.0+cu126, and an NVIDIA RTX 4070 Laptop GPU.

To reproduce the PyTorch installation on a compatible Windows/NVIDIA system:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

The other Python dependencies are listed in the repository's `requirements.txt`. Other hardware may require a different PyTorch build.

Run these checks from the repository root:

```powershell
python -m src.cnn_1D.cnn_smoke_test
python -m src.cnn_1D.cnn_handoff_check
```

- **Smoke test:** Verifies model dimensions, parameter count, loss, gradients, and checkpoint serialization.
- **Handoff check:** Reloads both saved checkpoints on CPU, applies their saved normalization, and reproduces validation metrics, including loss, accuracy, confusion matrices, and per-class results.

Both checks passed.


### Loading a Model for Inference

This is an example on how to load either checkpoint, apply its saved normalization, and generate predictions. This process can be reused when integrating the CNN for compression experiments.

```python
import numpy as np
import torch

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN

# 1. Load the checkpoint and reconstruct the model.
checkpoint = torch.load(
    "cnn_outputs/cnn_rough_seed42.pth",
    map_location="cpu",
    weights_only=True
)

model = BearingCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Prepare raw, unstandardized windows (N, 5000).
X_raw, _ = load_split("val")  # Replace with your own raw windows.

X = np.ascontiguousarray(
    ((X_raw - checkpoint["train_mean"]) /
     checkpoint["train_std"])[:, None, :],
    dtype=np.float32
)

# 3. Generate predictions.
with torch.inference_mode():
    logits = model(torch.from_numpy(X))

predictions = logits.argmax(dim=1).tolist()
class_names = checkpoint["class_names"]

print([class_names[i] for i in predictions[:5]])
```

Run the example from the repository root. Replace X_raw with your own raw vibration windows. To use the augmented model, change the checkpoint path to cnn_outputs/cnn_rough_aug_seed42.pth. Future experiments will save checkpoints under cnn_outputs/<run-folder>/checkpoint.pth. Keep model.eval() enabled for inference because BatchNorm and Dropout behave differently during training.

For compression experiments, the model follows this input/output contract:

| Property | Value |
|---|---|
| Input shape | `(batch, 1, 5000)` |
| Input dtype | `float32` |
| Output shape | `(batch, 3)` |
| Output type | Raw classification logits |
| Class order | Normal, InnerRaceFault, OuterRaceFault |
| Normalization | Saved clean-training mean and standard deviation |

Inputs must be standardized using the checkpoint's saved `train_mean` and `train_std` before inference. Do not standardize already normalized inputs again. Predicted class indices are obtained using `argmax(dim=1)` and mapped using the saved class order.

For the complete checkpoint verification process, see `cnn_handoff_check.py`, which reloads both models and evaluates their performance on the validation set.

Checkpoint loading and inference have been verified. Compatibility with the actual pruning, quantization, and projection pipelines has not yet been confirmed.

## 2. Dataset and Preprocessing

The CNN uses the provided MATLAB vibration dataset, loaded through the team's existing `src/data.py` utilities.

| Split | Windows | Per class |
|---|---:|---:|
| Training | 393 | 131 |
| Validation | 27 | 9 |
| Test | 102 | 34 |

Each window contains 5,000 vibration samples. The original splits are preserved.

Class mapping:

- `0` — Normal
- `1` — InnerRaceFault
- `2` — OuterRaceFault

### Global Standardization

The CNN uses global z-score standardization fitted exclusively on the clean training data, following the September standardization study.

Verified statistics:

```text
Training mean:           -0.1537873379955807
Training population std:  1.2870536466273184
```

These values are also stored in both checkpoints.

## 3. CNN Architecture

The model contains four convolutional blocks. Each block follows:

`Conv1D → BatchNorm1D → ReLU → MaxPool1D(4)`

| Block | Channels (in → out) | Kernel size | Output length |
|---|---|---:|---:|
| 1 | 1 → 16 | 15 | 1,250 |
| 2 | 16 → 32 | 9 | 312 |
| 3 | 32 → 64 | 7 | 78 |
| 4 | 64 → 96 | 5 | 19 |

The channel count increases across blocks to learn more feature representations, while max pooling progressively reduces the temporal dimensions.

The classifier uses:

`AdaptiveAvgPool1d(1) → Flatten → Dropout(0.20) → Linear(96, 3)`

**Total trainable parameters: 50,819.**

This is a compact starting architecture that has been trained successfully, but has not yet been hyperparameter-tuned or optimized.

## 4. Training

Both preliminary models used the same training configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 60 |
| Random seed | 42 |
| Loss | CrossEntropyLoss |
| Checkpoint selection | Lowest validation loss |

Training and validation histories are saved as CSV files.

### Training Augmentation

Two training versions were implemented:

- **Clean:** Uses the original training windows with global standardization.
- **Augmented:** Uses on-the-fly circular shifts (up to ±500 samples) and Gaussian noise (20 dB SNR). Each transformation has an independent 50% probability of being applied, meaning both can occur on the same window.

Augmentation happens before standardization and is applied only during training. Validation remains unaugmented, and training metrics are evaluated using clean windows.

The augmented dataset still processes 393 examples per epoch rather than expanding the dataset into additional fixed copies.

### Output Management

The original checkpoints and histories retain their filenames in `cnn_outputs/`.

Future training runs use configuration-specific subdirectories containing `checkpoint.pth` and `history.csv`. The `cnn_run_paths.py` helper prevents existing experiments from being silently overwritten.

## 5. Verification and Preliminary Results

The implementation passed architecture, gradient, checkpoint reload, and nine-example overfitting tests. Both full-data training runs completed successfully.

| Metric | Clean | Augmented |
|---|---:|---:|
| Selected epoch | 60 | 60 |
| Validation accuracy | 100% | 100% |
| Correct predictions | 27/27 | 27/27 |
| Validation loss (saved) | 0.000242 | 0.000159 |
| Macro-F1 | 1.00 | 1.00 |
| Checkpoint reload | Passed | Passed |

Both models produced the following validation confusion matrix:

```text
                 Predicted
                 N  IRF ORF
Actual Normal    9   0   0
Actual IRF       0   9   0
Actual ORF       0   0   9
```

Both checkpoints were successfully loaded into new models on CPU and reproduced their saved validation accuracy. The validation losses and raw model outputs (logits) also closely matched the saved results, allowing for small floating-point differences between CPU and GPU inference.

Both models achieved 100% accuracy on the supplied validation set. However, the validation set contains only 27 windows, and these results do not establish an augmentation benefit or generalization to independent recordings.

Final test-set evaluation and benchmarking are still pending.

## 6. Data Leakage and Limitations

Because both models reached 100% validation accuracy, two additional leakage checks were run.

**1. Exact duplicates and waveform correlation**

- No exact duplicate windows were found across the dataset splits.
- Maximum absolute train/validation correlation: **0.299428**
- Maximum absolute train/test correlation: **0.428588**
- No window pairs with absolute correlation ≥ 0.999 were found in those comparisons.

**2. Exact signal-segment overlap**

- Checked for identical 64-sample segments appearing at different positions across training and validation windows.
- No matching segments were found under the diagnostic procedure.

These checks did not identify evidence of direct duplication-based leakage.

However, **recording-level independence is still unverified**. It is unknown whether windows from the same original vibration recording were assigned to different splits. This could introduce shared recording characteristics even when windows are not exact duplicates or highly correlated.

Recording identifiers, window-overlap details, the split-generation procedure, and resampling details are also unavailable.

Additionally, the dataset comes from a single controlled laboratory test rig. Performance on these splits does not necessarily represent performance on independent recordings, different machines, or real-world operating conditions.

**The 100% validation accuracy should not be interpreted as proof that the model generalizes beyond the provided dataset splits.**

## 7. Files

All implementation files are located in `src/cnn_1D/`.

| File | Purpose |
|---|---|
| `cnn_model.py` | CNN architecture and class mapping |
| `cnn_train.py` | Clean baseline training |
| `cnn_train_aug.py` | Augmented baseline training |
| `cnn_augmentation.py` | Training-only augmentation |
| `cnn_smoke_test.py` | Architecture and gradient checks |
| `cnn_overfit_test.py` | Nine-example memorization test |
| `cnn_handoff_check.py` | Checkpoint loading and validation |
| `cnn_run_paths.py` | Experiment output management |

Saved checkpoints and training histories are located in `cnn_outputs/`.

The models are available for preliminary compression integration. Final tuning, three-seed evaluation, the full augmentation comparison, model-size/latency benchmarking, and MATLAB comparison remain pending.