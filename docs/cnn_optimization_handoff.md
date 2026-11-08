# 1D CNN Optimization — Final Technical Handoff

- **Project:** Break Through Tech AI × MathWorks
- **Date:** 2026-11-07
- **Branch:** `feature/1d-cnn-optimization`
**Status:** Individual 1D CNN optimization and evaluation complete

## 1. Objective and scope

Develop and optimize a 1D convolutional neural network for three-class bearing-fault classification using raw vibration windows.

The study compared architecture width, dropout, learning rate, training augmentation, validation robustness, serialized model size, and CPU inference latency.

The original CNN remains the team's compression reference. The micro CNN is the selected optimized dense alternative. This work does not replace teammates' pruning, quantization, projection, or other compression experiments.

## 2. Data and preprocessing

- Train: 393 windows, 131 per class
- Validation: 27 windows, 9 per class
- Test: 102 windows, 34 per class
- Window length: 5,000 signal samples
- Classes (fixed order): `Normal`, `InnerRaceFault`, `OuterRaceFault`
- Input tensor: `(batch, 1, 5000)`

Apply a global z-score using **clean-training-only** statistics stored in each checkpoint. Original training mean: approximately `-0.1537873379955807`; original population standard deviation: approximately `1.2870536466273184`. Never fit normalization statistics on validation or test data.

## 3. Experiment design

**Thirty distinct training runs** were completed:

- H1–H6: six registered hyperparameter configurations, three seeds each (18 runs)
- H7: post-hoc registered micro-width amendment, three seeds (3 additional runs)
- Four training augmentation modes, three seeds each; the three clean-original runs overlap with H1 (9 additional distinct runs)

Common training configuration: Adam optimizer; cross-entropy loss; batch size 32; learning rate 0.001 unless explicitly varied; 60 epochs; seeds 42, 1337, and 2026. Checkpoints were selected by **minimum clean-validation cross-entropy**.

Training augmentation conditions on the original width: none; circular shift up to ±500 samples with probability 0.5; Gaussian white noise at 20 dB SNR with probability 0.5; or both transforms independently applied. Augmentation was restricted to training data.

All 30 selected checkpoints achieved 100% accuracy on the same 27 clean-validation windows.

## 4. Architecture comparison

| Architecture | Channels | Parameters | Clean validation accuracy |
|---|---|---:|---:|
| Original | 16/32/64/96 | 50,819 | 100% |
| Medium | 12/24/48/72 | 28,803 | 100% |
| Small | 8/16/32/48 | 12,995 | 100% |
| Tiny | 4/8/16/24 | 3,395 | 100% |
| Micro | 2/4/8/12 | 923 | 100% |

Clean-validation accuracy did not distinguish the tested widths. Micro has **98.2% fewer trainable parameters** than the original, but **923 parameters is only the smallest configuration tested—not a demonstrated minimum**. The clean-validation capacity/accuracy failure point (the “knee”) was not reached; all tested widths remained at 100%.

## 5. Validation robustness

The fixed robustness suite contains **20 conditions**: the original clean validation set, 15 Gaussian-noise conditions (five SNR levels × three fixed realizations), and four circular-shift offsets (−500, −250, +250, +500). Each uses the same 27 original validation windows. Base perturbation seed: `20261107`; corruptions were applied to raw signals before fixed clean-training normalization.

Three-seed mean accuracy by width:

| Width | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|
| Original | 98.77% | 88.07% | 64.20% |
| Medium | 100.00% | 86.01% | 58.02% |
| Small | 100.00% | 97.12% | 67.90% |
| Tiny | 100.00% | 99.59% | 72.84% |
| Micro | 100.00% | 99.59% | 87.24% |

All tested widths achieved 100% accuracy under the four fixed circular shifts.

**Important distinction between the three-seed result and the shipped model:** Micro's headline **87.24% at 0 dB is the three-seed mean**, not the performance of its packaged checkpoint. Micro seed 42 achieved 92.59%, **selected seed 1337 achieved 69.14% (the weakest micro seed)**, and seed 2026 achieved 100%. Thus the micro model's severe-noise performance is **strongly seed-dependent** (69.14%–100%), and the shipped seed-1337 checkpoint performed only modestly better than the original seed-42 model's 66.67% at 0 dB. The packaged seed was chosen by clean-validation loss as specified in the selection procedure, **not** by robustness or final test results; do not switch seeds after reviewing these outcomes.

For the original architecture, augmentation modes yielded these three-seed mean accuracies at 0 dB:

- None: 64.20%
- Shift: 66.67%
- Noise: 61.73%
- Both: 65.02%

**No consistent augmentation advantage was observed** across the tested noise levels. Because perturbations reuse the same 27 original windows, the 243 model–noise observations at a given SNR are correlated rather than 243 independent samples. No formal statistical test established an effect or its absence; the differences should be treated as descriptive.

These are synthetic validation stress tests, not estimates of unseen real-world operating conditions.

## 6. Frozen final test evaluation

The candidate groups and three-seed analysis plan were frozen in `docs/cnn_final_selection_freeze.md` before formal final test evaluation. The **six frozen groups** were:

1. Original clean CNN (no augmentation, dropout 0.20)
2. Original with shift-only training augmentation
3. Original with noise-only training augmentation
4. Original with combined shift-and-noise training augmentation
5. Original clean CNN with dropout 0.00
6. Micro clean CNN (channels 2/4/8/12, dropout 0.20)

Each was evaluated at seeds 42, 1337, and 2026 (**18 existing checkpoints**). All 18 achieved 102/102 test accuracy (100%) and macro-F1 of 1.00. Per-seed confusion matrices and per-class precision, recall, and F1 are recorded in `cnn_outputs/final_eval/frozen_test_results_v1.json`. Medium, small, and tiny widths were **not** included in the frozen test evaluation. No post-test tuning or additional model selection was performed.

**Uncertainty:** For one fixed model with 102 correct predictions out of 102, the nominal two-sided 95% exact binomial (Clopper–Pearson) lower confidence bound is about **96.45%**, *if* windows are independent, representative draws from the target distribution. That assumption is unverified here; shared recording/session structure could invalidate this bound. The 18 evaluations also reuse the same 102 windows, not 18 independent test sets. Perfect observed accuracy does **not** establish perfect population accuracy.

## 7. Final packaged models and definition of size

### Original compression reference

Artifact: `cnn_outputs/final_models/original_reference_seed42.pth`

- Seed: 42
- Channels: `(16, 32, 64, 96)`
- Trainable parameters: 50,819
- Serialized inference-package size: **0.214793 MB**
- Test accuracy: 100%
- Role: established team compression reference

### Optimized dense alternative

Artifact: `cnn_outputs/final_models/micro_selected_seed1337.pth`

- Seed: 1337
- Channels: `(2, 4, 8, 12)`
- Trainable parameters: 923
- Serialized inference-package size: **0.013641 MB**
- Test accuracy: 100%
- Selection criterion: lowest clean-validation loss among the three micro seeds (`0.005288184`)
- **0 dB robustness of this specific checkpoint: 69.14%**, not the three-seed mean of 87.24%

Both inference packages contain model weights, BatchNorm state/buffers, architecture configuration, class labels, and training normalization statistics. The sizes above are **decimal megabytes of the actual `.pth` files on disk**, measured via the shared `measure_model_size_mb(path=...)` function—not parameter counts or compressed sparse-weight sizes.

At float32, trainable parameter tensors alone account for approximately **203,276 bytes** for the original and **3,692 bytes** for micro (about 55× fewer parameter bytes). Actual package sizes are **214,793** and **13,641 bytes**, respectively: they also include non-parameter buffers, metadata, and serialization overhead. Hence the reduction is **98.2% in parameters** but **93.6% in shipped file size**.

Hashes and source-checkpoint provenance are in `cnn_outputs/final_models/manifest.json`.

## 8. Official CPU benchmarking

Both final models were measured in the **same Google Colab CPU session**, using one PyTorch thread, batch size 1, 10 warmup calls, and 100 timed calls per model. The input is a raw 5,000-sample validation window; the timed prediction function includes global normalization and tensor conversion in addition to PyTorch inference.

| Metric | Original seed 42 | Micro seed 1337 |
|---|---:|---:|
| Inference-package size (MB) | 0.214793 | 0.013641 |
| Mean CPU latency (ms) | 1.250706 | 0.723028 |
| CPU latency SD across 100 calls (ms) | 0.048445 | 0.265935 |

In **this session**, micro's mean was **1.73× faster** (42.2% less mean latency) than the original. These are descriptive same-session measurements, not stable guarantees across Colab hosts or embedded hardware. Micro's per-call variability was appreciable. The report contains means and standard deviations, **not medians or per-call samples**; a median cannot be recovered. The elapsed time was not decomposed into preprocessing, Python/tensor overhead, and model computation, so we cannot attribute a specific part of the micro model's ~0.72 ms to any one stage. A second matched session would strengthen a general speed claim, but is not required to report the observed session comparison honestly.

Benchmark record: `cnn_outputs/benchmarks/final_colab_cpu_v1.json`

Re-run from repository root in a CPU environment with dependencies installed:

```bash
python -m src.cnn_1D.cnn_final_team_benchmark
```

For official comparisons with other methods, time their deployment artifacts alongside these baselines in the same CPU session. Colab CPU latency does not establish latency on MathWorks-specific embedded hardware.

## 9. Team integration

`results/experiments.csv` contains the following new entries:

- **Experiment 1:** original 1D CNN, seed 42
- **Experiment 2:** micro 1D CNN, seed 1337

Both have `Compression = None`: channel-width reduction is a dense architecture change, not weight pruning or quantization. The original remains available as a compression reference; micro is a separate dense deployment alternative, not a replacement for existing compression experiments.

## 10. Important limitations

1. The original validation split contains only 27 windows.
2. All windows come from a controlled bearing dataset.
3. Recording/session-level independence remains unverified.
4. Duplicate and overlap diagnostics did not establish full recording-level independence.
5. The team's test split has previously been inspected; the evaluation is not completely blind.
6. Perfect accuracy on this split does not establish generalization to new machines, sensors, or conditions.
7. Synthetic Gaussian-noise robustness does not establish operational robustness in the field.
8. Colab CPU latency is not embedded-device latency.
9. Differences across only three training seeds do not establish a general causal relationship between model width and robustness.

## 11. Artifact availability and reproducibility

**Committed:** experiment protocols and selection freeze; training/evaluation scripts; augmentation and width robustness CSVs; frozen test JSON; dense CPU benchmark CSV; official Colab CPU benchmark JSON; two inference packages and hash manifest; team experiment-tracker entries.

**Retained locally:** additional training checkpoints and per-epoch histories; deterministic validation perturbation `.npz`. The perturbation dataset can be regenerated using the committed deterministic generator. A fresh clone can load the two packaged models, but rerunning every historical checkpoint evaluation requires regenerating or separately sharing the uncommitted training artifacts. Preserve local experiment artifacts for auditing; do not delete them as part of Git cleanup.

## 12. Final outcome

The **923-parameter micro CNN** matched the original **50,819-parameter CNN** on the supplied clean test split, while reducing inference-package size by **93.6%** and measured mean Colab CPU latency by **42.2% in one matched session**. Its apparent severe-noise advantage is **not** reliably represented by the selected seed-1337 artifact: the 87.24% micro result is a three-seed mean, whereas the packaged seed achieved 69.14% at 0 dB.

The original remains the team's compression reference. Micro is a compact dense alternative; team-level deployment selection remains a separate decision.
