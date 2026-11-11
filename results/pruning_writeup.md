# Task #4: Model Pruning: Results and Write-up

Aigerim Zhadikbay · **Issue:** [#17]

## What this task did

Applied structured and unstructured pruning to the team's 1D CNN baseline at
25%, 50%, 75%, and 90% weight removal (8 experiments total), fine-tuned each
pruned model briefly, and measured test accuracy, file size, **zipped** file
size, and inference latency for every variant. All results are logged as rows
1–9 of `results/experiments.csv` and charted in
`results/pruning_accuracy_vs_level.png`.

**Baseline used:** Gurbaj's rough clean checkpoint
(`cnn_outputs/cnn_rough_seed42.pth`, 50,819 parameters, 100% val/test accuracy).
The final frozen baseline was not available by the experiment deadline, so per
the Part 1 plan everything ran against the rough model. If a final baseline
lands, every experiment can be re-run with one flag (`--checkpoint <path>`).

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

## Results

Measured in one session on one machine (MacBook Air, CPU, 1 thread), so the
rows are directly comparable. The dense baseline was re-measured in the same
session (row 1 of the CSV).

| Model | Test accuracy | .pth size (MB) | Zipped (MB) | Latency (ms/sample) |
|---|---:|---:|---:|---:|
| Dense baseline | 100.0% | 0.213 | 0.193 | 0.49 |
| Unstructured 25% | 100.0% | 0.213 | 0.160 | 0.54 |
| Unstructured 50% | 100.0% | 0.213 | 0.119 | 0.51 |
| Unstructured 75% | 100.0% | 0.213 | 0.075 | 0.50 |
| **Unstructured 90%** | **100.0%** | **0.213** | **0.045** | **0.51** |
| Structured 25% | 100.0% | 0.213 | 0.153 | 0.46 |
| Structured 50% | 100.0% | 0.213 | 0.111 | 0.45 |
| Structured 75% | 81.4% | 0.213 | 0.068 | 0.39 |
| Structured 90% | 47.1% | 0.213 | 0.041 | 0.34 |

Fine-tuning mattered: before it, unstructured 75%/90% had fallen to 59%/67%
validation accuracy and structured 50–90% to 33% (random chance). Fine-tuning
recovered everything except structured 75% (→ 81%) and structured 90% (→ 48%).

## Pushing unstructured past 90%: the break point

Unstructured pruning was still at 100% at 90%, so the grid was extended until
accuracy fell off (same pipeline: 15 fine-tune epochs at lr=1e-4).

| Model | Weights left | Test accuracy | Val accuracy | Zipped (MB) |
|---|---:|---:|---:|---:|
| Unstructured 92% | ~4,000 | 100.0% | 100.0% | 0.041 |
| Unstructured 94% | ~3,000 | 100.0% | 100.0% | 0.037 |
| Unstructured 95% | ~2,500 | 100.0% | 100.0% | 0.034 |
| Unstructured 96% | ~2,000 | 100.0% | 100.0% | 0.032 |
| **Unstructured 97%** | **~1,500** | **100.0%** | **100.0%** | **0.030** |
| Unstructured 98% | ~1,000 | 97.1% | 88.9% | 0.027 |
| Unstructured 99% | ~500 | 37.3% | 48.1% | 0.024 |
| Unstructured 99.5% | ~250 | 33.3% | 33.3% | 0.023 |

("Weights left" counts the ~50,200 prunable Conv1d/Linear weights.)

**The break point is between 97% and 98%.** Accuracy holds at 100% through
97%, starts slipping at 98%, and collapses to chance at 99%. The margin at 97%
is thin: its validation loss was 0.11 (vs 0.0007 at 92%) and was still falling
at the last fine-tune epoch, so 97% sits right at the edge. Longer fine-tuning
might push the break point slightly further.

These rows are ids 10–17 in `experiments.csv`. Their latency was measured in a
later session where the dense baseline itself ran at 1.13 ms, so latency is
only comparable within that group (it was flat, ~1.0–1.15 ms, at every level).

## Getting around "the zeros still take space"

Mask-based pruning keeps every weight as a 4-byte float, zero or not. To turn
the zeros into real savings, the model has to be stored or rebuilt
differently. Measured on the saved checkpoints:

| Storage format | Dense baseline | Unstr. 90% | Unstr. 97% |
|---|---:|---:|---:|
| Plain `.pth` (current) | 0.214 MB | 0.214 MB | 0.214 MB |
| Zipped `.pth` (current) | 0.193 MB | 0.045 MB | 0.030 MB |
| PyTorch sparse COO | 1.432 MB | 0.168 MB | 0.070 MB |
| PyTorch sparse CSR | 0.620 MB | 0.077 MB | 0.035 MB |
| Nonzero values + 1-bit mask | 0.227 MB | 0.046 MB | 0.032 MB |
| Nonzero values + 1-bit mask, zipped | 0.194 MB | **0.030 MB** | **0.015 MB** |

- **PyTorch's built-in sparse formats are a trap at this size.** They store a
  64-bit index for every nonzero weight, so they are only smaller than dense at
  very high sparsity, and never beat simply zipping.
- **Storing only the nonzero values plus a 1-bit-per-weight mask** (which
  weights survived) and zipping it is the smallest option: 0.030 MB at 90%
  (~7× smaller than the dense `.pth`) and 0.015 MB at 97% (~14×). The model is
  unpacked back to dense to run, so this saves storage/transfer, not compute.
- **Physically removing structured-pruned filters** is the only route that
  also makes the model faster: rebuild a narrower `BearingCNN` without the dead
  filters. For structured 50% (which kept 100% accuracy) the layers become
  8/16/32/48 channels instead of 16/32/64/96, about 13,200 parameters instead of
  51,235, so a raw `.pth` around 4× smaller and far fewer multiplications. Not
  implemented yet; note that a dead conv filter is followed by BatchNorm, which
  turns its zero output into a constant, so that constant has to be folded into
  the next layer's bias when the filter is removed.
- **Quantization** (float32 → int8) shrinks every remaining weight 4×, and
  stacks with either option above. That is the combined-compression step.

## Trade-off summary

Unstructured pruning was essentially free on this task: even with 90% of
weights removed, the model still classified all 102 test windows correctly,
which says the baseline CNN is heavily over-parameterized for this dataset.
Structured pruning was much more damaging — fine to 50%, then 81% accuracy at
75% and 47% at 90% — because it deletes whole feature detectors rather than
letting the pruning spread across redundant weights. The counterintuitive
headline: **pruning did not shrink the saved file at all** — a zeroed float32
weight occupies the same 4 bytes as any other, so the `.pth` stayed at 0.213 MB
at every level; only the *zipped* size shrinks (0.193 → 0.045 MB, ~4.3×),
because runs of zeros compress well. Latency also did not improve, since dense
hardware multiplies zeros at full cost. Structured pruning is the only style
whose zeros could be converted into *real* raw-size and speed wins — by
physically rebuilding the network without the dead filters — but that costs
accuracy, whereas unstructured sparsity can never be cashed in on standard
hardware.

## Best model (handoff)

**`cnn_outputs/pruned/unstructured_90pct/checkpoint.pth`** — it dominates every
other variant: 100% test accuracy at the smallest zipped size. It loads exactly
like the baseline (see the loading example in `src/cnn_1D/README.md`); the
checkpoint also stores its pruning settings and measured results. Recommended
next step for the combined-compression experiment: quantize this model.

## Caveats

- The test set has 102 windows, so accuracy resolution is ~1%; identical 100%
  scores at 25–90% unstructured mean the dataset can't distinguish those
  models, not that pruning costs nothing. The unstructured "cliff" presumably
  exists somewhere above 90% — untested.
- Results are from the *rough* baseline, single seed (42), single machine.
- Robustness of pruned models under noise is untested (relevant to Farheen's
  noise stress-test task — these checkpoints are ready inputs for it).

## Reproducing

```bash
# one experiment (repeat for each method × level):
python3 -m src.cnn_1D.cnn_prune --method unstructured --level 0.9 --eval-test

# re-measure everything and append rows to results/experiments.csv:
python3 -m src.cnn_1D.log_pruning_results --save

# regenerate the chart:
python3 results/pruning_plot.py
```

Files added by this task: `src/cnn_1D/cnn_prune.py` (pruning pipeline),
`src/cnn_1D/log_pruning_results.py` (CSV logging),
`results/pruning_plot.py` (chart), `results/pruning_accuracy_vs_level.png`,
rows 1–9 of `results/experiments.csv`, and the pruned checkpoints under
`cnn_outputs/pruned/`.
