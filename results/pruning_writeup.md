# Task #4: Model Pruning: Results and Write-up

Aigerim Zhadikbay · **Issue:** [#17]

## What this task did

Applied structured and unstructured pruning to the team's 1D CNN at 25%, 50%,
75%, and 90% weight removal, then pushed unstructured pruning further (92% to
99.5%) to find where accuracy breaks. Each pruned model was fine-tuned briefly,
saved in a sparse format that only stores the surviving weights, and measured
for test accuracy, shipped file size, and inference latency. All results are
logged as rows 1–17 of `results/experiments.csv` (following
`measurement-guide.md`) and charted in `results/pruning_accuracy_vs_level.png`.

**Model pruned:** `cnn_outputs/cnn_rough_seed42.pth` (50,819 parameters, 100%
val/test accuracy). This is the same model as the team's final compression
reference, `cnn_outputs/final_models/original_reference_seed42.pth` on
`feature/1d-cnn-optimization`: that package was built from this checkpoint
(matching SHA-256 in its manifest) and has identical weights.

## The two pruning styles

| | Unstructured | Structured |
|---|---|---|
| What gets removed | Individual weights, chosen globally by smallest magnitude (L1) across all conv + linear layers | Whole conv filters (output channels), chosen per layer by smallest L2 norm; filter biases are masked too so pruned channels are fully dead |
| Where it can cut | Anywhere it finds redundancy | Entire feature detectors; the final `Linear(96, 3)` is excluded so no output class is deleted |
| Implementation | `torch.nn.utils.prune.global_unstructured` | `torch.nn.utils.prune.ln_structured(dim=0)` + bias masks |

After pruning, each model was fine-tuned for 15 epochs at lr=1e-4 on the clean
training split with the pruning masks pinned (zeros cannot revive), selecting
the lowest-validation-loss epoch — the same selection rule as baseline
training. The test set was never used for any decision, only for final scoring.

## How each model is saved and measured

Pruning with masks only sets weights to zero; the tensors keep their shape, so
a normal `.pth` stays the same size at every pruning level (~0.215 MB). Per the
measurement guide ("size is the file you would actually ship"), each pruned
model is also saved as **`model_sparse.npz`**
(`src/cnn_1D/cnn_sparse.py`): the nonzero float32 weights, a 1-bit-per-weight
mask saying which positions were kept, and the tensor shapes, written with
`np.savez_compressed`. Loading it rebuilds the exact dense model: weights and
test-set outputs match the original checkpoint bit for bit for all 17 models.

- **Size** = the sparse file, via `measure_model_size_mb(path=...)`. The dense
  reference row uses its normal `.pth`.
- **Accuracy and latency** come from the model loaded back from the sparse
  file, so they describe what would actually be shipped.
- **Latency**: batch size 1, one thread, shared `measure_inference_latency_ms`
  defaults, with the dense reference re-timed in the same run as every pruned
  model. Timed on a MacBook (macOS, x86_64), **not yet on Colab CPU** as the
  guide requires; `log_pruning_results.py` records the platform in `notes` and
  should be re-run on Colab for official latency.

## Results

| Model | Test accuracy | Shipped size (MB) | Smaller than dense | Latency (ms/sample) |
|---|---:|---:|---:|---:|
| Dense reference (`.pth`) | 100.0% | 0.2165 | 1.0× | 1.08 |
| Unstructured 25% | 100.0% | 0.1478 | 1.5× | 1.05 |
| Unstructured 50% | 100.0% | 0.1008 | 2.1× | 1.05 |
| Unstructured 75% | 100.0% | 0.0547 | 4.0× | 1.02 |
| **Unstructured 90%** | **100.0%** | **0.0262** | **8.3×** | **1.02** |
| Unstructured 92% | 100.0% | 0.0222 | 9.7× | 1.01 |
| Unstructured 94% | 100.0% | 0.0183 | 11.9× | 1.01 |
| Unstructured 95% | 100.0% | 0.0162 | 13.4× | 1.02 |
| Unstructured 96% | 100.0% | 0.0141 | 15.3× | 1.01 |
| **Unstructured 97%** | **100.0%** | **0.0120** | **18.0×** | **1.01** |
| Unstructured 98% | 97.1% | 0.0098 | 22.0× | 1.02 |
| Unstructured 99% | 37.3% | 0.0075 | 29.0× | 0.99 |
| Unstructured 99.5% | 33.3% | 0.0062 | 34.9× | 0.92 |
| Structured 25% | 100.0% | 0.1448 | 1.5× | 0.97 |
| Structured 50% | 100.0% | 0.0984 | 2.2× | 0.91 |
| Structured 75% | 81.4% | 0.0520 | 4.2× | 0.87 |
| Structured 90% | 47.1% | 0.0242 | 8.9× | 0.84 |

Fine-tuning mattered: before it, unstructured 75%/90% had fallen to 59%/67%
validation accuracy and structured 50–90% to 33% (random chance). Fine-tuning
recovered everything except structured 75% (→ 81%) and structured 90% (→ 48%).

## The unstructured break point

Unstructured pruning was still at 100% at 90%, so it was pushed until accuracy
fell off:

| Removed | Weights left | Test accuracy | Val accuracy |
|---:|---:|---:|---:|
| 92% | ~4,000 | 100.0% | 100.0% |
| 95% | ~2,500 | 100.0% | 100.0% |
| **97%** | **~1,500** | **100.0%** | **100.0%** |
| 98% | ~1,000 | 97.1% | 88.9% |
| 99% | ~500 | 37.3% | 48.1% |
| 99.5% | ~250 | 33.3% | 33.3% |

("Weights left" counts the ~50,200 prunable Conv1d/Linear weights.)

**The break point is between 97% and 98%.** Accuracy holds at 100% through
97%, starts slipping at 98%, and collapses to chance at 99%. The margin at 97%
is thin: its validation loss was 0.11 (vs 0.0007 at 92%) and was still falling
at the last fine-tune epoch, so 97% sits right at the edge. Longer fine-tuning
might push the break point slightly further.

## Other ways to store the zeros (compared)

Measured on the same checkpoints before choosing the sparse format:

| Storage format | Dense | Unstr. 90% | Unstr. 97% |
|---|---:|---:|---:|
| Plain `.pth` | 0.214 MB | 0.214 MB | 0.214 MB |
| Zipped `.pth` | 0.193 MB | 0.045 MB | 0.030 MB |
| PyTorch sparse COO | 1.432 MB | 0.168 MB | 0.070 MB |
| PyTorch sparse CSR | 0.620 MB | 0.077 MB | 0.035 MB |
| **`model_sparse.npz` (used)** | 0.191 MB | **0.026 MB** | **0.012 MB** |

- **PyTorch's built-in sparse formats don't help at this size.** They store a
  64-bit index for every nonzero weight, so they only beat dense at very high
  sparsity and never beat simply zipping.
- **Nonzero values + 1-bit mask** is the smallest. Packing all tensors into
  three arrays matters: one archive entry per tensor cost ~18 KB of headers,
  which at 90% pruning was more than the surviving weights themselves.
- **This saves storage, not compute.** The model is unpacked back to dense to
  run, so it is just as slow.
- **Physically removing structured-pruned filters** is the only route that
  would also make the model faster: rebuild a narrower `BearingCNN` without the
  dead filters. For structured 50% (100% accuracy) the layers would become
  8/16/32/48 channels instead of 16/32/64/96 — the same widths as the "small"
  CNN on `feature/1d-cnn-optimization` (12,995 parameters). Not implemented; a
  dead conv filter is followed by BatchNorm, which turns its zero output into a
  constant, so that constant has to be folded into the next layer's bias.
- **Quantization** (float32 → int8) shrinks every remaining weight 4× and
  stacks with the sparse format. That is the combined-compression step.

## Trade-off summary

Unstructured pruning was essentially free on this task up to 97%: with only
~1,500 of ~50,000 weights left, the model still classified all 102 test windows
correctly, which says the CNN is heavily over-parameterized for this dataset.
Structured pruning was much more damaging — fine to 50%, then 81% accuracy at
75% and 47% at 90% — because it deletes whole feature detectors rather than
letting the pruning spread across redundant weights.

Pruning alone does not shrink a normal `.pth` file, because a zeroed float32
weight still takes 4 bytes. Saving only the surviving weights does: the 90%
model ships at 0.026 MB (8.3× smaller) and the 97% model at 0.012 MB (18×).
Latency did not improve for unstructured pruning, since the model is run dense
and the zeros are still multiplied. Structured pruning timed somewhat faster at
high levels (0.84 vs 1.08 ms at 90%) even though the layer shapes are
unchanged; the cause was not investigated (one possibility: dead channels
produce exact zeros, avoiding slow very-small-number arithmetic), so it should
not be read as a reliable speed-up.

For comparison, the team's dense **micro CNN** (923 parameters, channels
2/4/8/12, 0.0136 MB, 100% test accuracy) reaches a similar size by training a
much narrower network from scratch, and it is actually faster. Pruning shows
the same thing from the other direction: the original network carries far more
weights than this task needs.

## Best model (handoff)

**`cnn_outputs/pruned/unstructured_90pct/`** (`checkpoint.pth`, plus
`model_sparse.npz` as its shipped form) remains the handoff for combined
compression: 100% test accuracy at 8.3× smaller, well away from the break
point. The 97% model is
smaller still at 100%, but it sits right at the edge, so it is a riskier input
for quantization, which adds its own error. `checkpoint.pth` loads exactly like
the baseline (see `src/cnn_1D/README.md`); `model_sparse.npz` loads with
`load_sparse` from `src/cnn_1D/cnn_sparse.py`.

## Caveats

- The test set has 102 windows, so accuracy resolution is ~1%; identical 100%
  scores mean the dataset can't distinguish those models, not that pruning
  costs nothing.
- Single seed (42), single fine-tuning schedule (15 epochs, untuned).
- Latency was timed on a MacBook, not Colab CPU; re-run the logger on Colab for
  the official numbers.
- Robustness of pruned models under noise is untested (relevant to the noise
  stress-test work — these checkpoints are ready inputs for it).

## Reproducing

```bash
# one experiment (writes checkpoint.pth and model_sparse.npz):
python3 -m src.cnn_1D.cnn_prune --method unstructured --level 0.9 --eval-test

# re-measure everything in one session and replace rows in experiments.csv:
python3 -m src.cnn_1D.log_pruning_results --save

# regenerate the chart:
python3 results/pruning_plot.py
```

Files added by this task: `src/cnn_1D/cnn_prune.py` (pruning pipeline),
`src/cnn_1D/cnn_sparse.py` (sparse save/load),
`src/cnn_1D/log_pruning_results.py` (CSV logging),
`results/pruning_plot.py` (chart), `results/pruning_accuracy_vs_level.png`,
rows 1–17 of `results/experiments.csv`, and the pruned checkpoints under
`cnn_outputs/pruned/`.
