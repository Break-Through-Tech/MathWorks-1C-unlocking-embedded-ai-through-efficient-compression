# Combined Compression and Robustness: Results and Write-up

Aigerim Zhadikbay · Issue #30

**Frozen checkpoint:** `cnn_rough_seed42` has identical weights to `original_reference_seed42`; no rerun needed.

## What was built

| Model | Compression (CSV) | Shipped file | Size (MB) | vs reference (0.214793 MB) | Test acc |
|---|---|---|---:|---:|---:|
| Pruned 97% + INT8 | Pruning + Quantization | `cnn_outputs/combined/pruned97_int8/model_sparse_int8.npz` | 0.005201 | 41.3x | 100% |
| Projected 10% (augmented) + INT8 | Projection + Quantization | `cnn_outputs/combined/projected10_int8/model_int8.pth` | 0.021218 | 10.1x | 100% |
| Micro + INT8 | Quantization | `cnn_outputs/combined/micro_int8/model_int8.pth` | 0.007906 | 27.2x | 100% |

Logged as `results/experiments.csv` IDs 38-40. Latency there came from a local
Mac CPU and is a placeholder for Tule's one-session EC2 benchmark.

- **Pruning + quantization.** The best pruned model at 100% test accuracy is
  unstructured 97% (`cnn_outputs/pruned/unstructured_97pct`). It is quantized
  with Venya's `QuantizableBearingCNN` (calibrated on the train split). A plain
  INT8 file would be 0.0593 MB because it stores the zeros too. The shipped file
  stores only the nonzero int8 weights plus a 1-bit mask, scales and float
  biases (`save_sparse_int8` in `src/cnn_1D/cnn_sparse.py`). Folding BatchNorm
  and symmetric int8 both keep zeros at zero, so 97.0% of the int8 weights are
  still zero and the mask is exact. Reloading gives identical logits.
- **Projection + quantization.** The 10%-rank augmented projected model with
  the fuse groups from `projection_writeup.md`. Predictions match the FP32
  projected model on all 102 test clips.
- **Micro model.** Quantizing it works (100% val/test, identical predictions),
  so it is included. The saving is modest in absolute terms (13.6 KB to 7.9 KB).

Code: `src/cnn_1D/combine_prune_quantize.py`, `combine_projection_quantize.py`,
`quantize_micro.py`, `combined_robustness.py`.

## Robustness (validation, Gurbaj's suite)

Mean accuracy over 3 noise draws on the 27 validation windows
(`results/combined_robustness.csv`; the dense reference reproduces Gurbaj's
recorded numbers exactly). All models score 100% on all four time shifts.

| Model | Size (MB) | Clean | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense reference | 0.2148 | 100% | 100% | 100% | 100% | 99% | 67% |
| Pruned 90% | 0.0262 | 100% | 100% | 100% | 100% | 84% | 67% |
| Pruned 95% | 0.0162 | 100% | 100% | 100% | 100% | 59% | 67% |
| Pruned 97% | 0.0120 | 100% | 100% | 100% | 96% | 69% | 51% |
| INT8 (Venya) | 0.0593 | 100% | 100% | 100% | 100% | 99% | 67% |
| Pruned 97% + INT8 | 0.0052 | 100% | 100% | 100% | 96% | 60% | 42% |
| Projected 10% + INT8 | 0.0212 | 100% | 100% | 100% | 100% | 95% | 64% |
| Micro | 0.0136 | 100% | 100% | 100% | 100% | 99% | 69% |
| Micro + INT8 | 0.0079 | 100% | 100% | 100% | 100% | 98% | 69% |

**Clean vs augmented:** already answered for projection (augmented 95% vs clean
83% at 5 dB at 10% rank, `results/projection_robustness.csv`); the augmented
model is the one combined here.

## Does combining help, and where does each method fail?

Combining helps size a lot: pruning + INT8 is 41x smaller than the reference
and projection + INT8 is 10x smaller, both at 100% clean test accuracy, and INT8
adds a 2.3x saving on top of pruning only because the sparse INT8 file stores
the mask. It costs robustness: INT8 alone and micro hold 98-99% at 5 dB, projected
+ INT8 starts to slip at 5 dB (95%), while pruned 97% and pruned 97% + INT8 start
failing already at 10 dB (96%) and fall to 69% and 60% at 5 dB. At 0 dB the
signal is as strong as the noise and everything is around 65% (the combined
pruned model drops to 42%). Time shifts of up to 500 samples never hurt any
model. Caveats: only 27 validation windows, synthetic noise, one seed per model,
and the 5 dB numbers for the 90/95/97% pruned models are not monotonic, so
treat the exact values as noisy.

## Loading the files (for Tule)

Run from the repository root. Normalization is the shared train mean/std (the
`meta` / package dict also holds `train_mean` and `train_std`). INT8 models use
the qnnpack engine, like Venya's INT8 file.

```python
torch.backends.quantized.engine = "qnnpack"

# Pruned 97% + INT8
from src.cnn_1D.cnn_sparse import load_sparse_int8
model, meta = load_sparse_int8("cnn_outputs/combined/pruned97_int8/model_sparse_int8.npz")

# Projected 10% + INT8, and Micro + INT8
from src.cnn_1D.combine_projection_quantize import load_projected_int8
model, package = load_projected_int8("cnn_outputs/combined/projected10_int8/model_int8.pth")
model, package = load_projected_int8("cnn_outputs/combined/micro_int8/model_int8.pth")

# All three: input is the standardized window (N, 1, 5000), float32
pred = model(x).argmax(dim=1)
```

Measure size with `measure_model_size_mb(path=...)` on the file named above.
