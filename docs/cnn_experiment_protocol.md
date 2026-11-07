# MathWorks 1D CNN — Registered Experiment Protocol

**Project:** Break Through Tech AI Studio × MathWorks, bearing-fault classification  
**Prepared:** 2026-11-07  
**Status:** READY FOR PROTOCOL-ONLY COMMIT — local audit completed; remaining runs not started  
**Branch:** `feature/1d-cnn-optimization` (local)  
**Scope:** Six-configuration hyperparameter study; four-way augmentation study; validation-only robustness; frozen final test evaluation; batch-one CPU benchmarking.

## 1. Provenance and prior work (do not present this as prospective)

The following work happened **before registration of this protocol**:

- Original clean 1D CNN: 16/32/64/96 channels; 50,819 trainable parameters; dropout 0.20; Adam LR 0.001; batch size 32; seed 42; 60 epochs; lowest-validation-loss checkpoint selected at epoch 60; validation accuracy 27/27; best validation loss approximately 0.000241749. Checkpoint `cnn_outputs/cnn_rough_seed42.pth`.
- Original augmented CNN: same architecture and optimizer; train-only on-the-fly circular shift with probability 0.5 (up to ±500 samples), noise with probability 0.5 (20 dB SNR), independently applied; seed 42; 60 epochs; validation accuracy 27/27; best validation loss approximately 0.000159242; checkpoint `cnn_outputs/cnn_rough_aug_seed42.pth`.
- Medium clean CNN: channels 12/24/48/72; 28,803 parameters; seed 42; LR 0.001; dropout 0.20; 60 epochs; validation accuracy 27/27; selected epoch 57; verified best validation loss `0.00020690350275900628` (final-epoch loss `0.00022209040751611746`). Local checkpoint `cnn_outputs/cnn_none_seed42_adam_lr0p001_bs32_ep60_m292b341a37_oe1f37b6114/checkpoint.pth`.
- Small clean CNN: channels 8/16/32/48; 12,995 parameters; seed 42; LR 0.001; dropout 0.20; 60 epochs; validation accuracy 27/27; selected epoch 60; verified best validation loss `0.0008021762801541223`. Local checkpoint `cnn_outputs/cnn_none_seed42_adam_lr0p001_bs32_ep60_m1c113ae8e7_oe1f37b6114/checkpoint.pth`.
- The original checkpoints passed smoke tests, CPU checkpoint reload, reproducibility checks, and validation confusion matrices. After merging `origin/main` into the new local optimization branch, the verification tests passed again. The two modified-width runs also reported checkpoint reload success.

These four prior runs are **retrospectively included**, not falsely described as having been preregistered. The November 7 local read-only audit confirmed checkpoint metadata, 60-row histories, and best-epoch selection for all four. The original two checkpoints do not explicitly store `channels` or `dropout`; their 16/32/64/96, 0.20 settings follow from the original verified baseline code, architecture smoke test and recorded training context. Do not overwrite or rerun working checkpoints solely to standardize their filenames.

## 2. Data, preprocessing, and scientific limitations

- Use original fixed splits: training 393 (131/class); validation 27 (9/class); test 102 (34/class); every window 5,000 signal samples.
- Classes and index order: `Normal`, `InnerRaceFault`, `OuterRaceFault`.
- Fit one global mean and population standard deviation to the **clean training split only**; apply those same statistics to validation, test, and train augmentations. Expected training mean approximately `-0.1537873379955807`, population standard deviation `1.2870536466273184`.
- No re-splitting and no normalization using validation or test statistics.
- Exact window-duplicate, high-correlation, and 64-sample segment-overlap checks found no direct duplication, but **recording/session-level independence remains unverified**. Dataset is from a single controlled test rig. Neither 100% validation accuracy nor a later 100% test result proves cross-recording or field generalization.
- Several teammates have already viewed test scores on the supplied split. The final study must not represent its evaluation as a completely blind, never-examined team test.

## 3. Common training and checkpoint protocol

- PyTorch `BearingCNN`, original four blocks with kernel sizes 15/9/7/5, BatchNorm1d, ReLU and MaxPool1d(4), followed by AdaptiveAvgPool1d(1), Flatten, configured Dropout, and a three-class linear output.
- Unless a registered row explicitly differs: Adam optimizer; initial LR `0.001`; batch size `32`; cross-entropy loss; **60 epochs, no early stopping**; original kernel sizes/pooling; same clean training/validation splits.
- Seeds for each configuration: **42, 1337, 2026**. Seed Python/NumPy/Torch/CUDA and the PyTorch DataLoader generator reproducibly; log versions, hardware and actual deterministic settings. Reproducible does not imply byte-for-byte cross-device determinism. Runs on other hardware are compared for accuracy/size, not unqualified CPU latency.
- After each epoch, evaluate clean, nonaugmented training and validation data. Select the checkpoint at the **lowest clean validation cross-entropy**, not the final epoch or test accuracy. Record the selected epoch, accuracy, and loss. Preserve complete per-epoch CSV history.
- Keep the 60-epoch budget unchanged for the controlled matrix: original clean and augmented selected epoch 60; medium selected 57; small selected 60. A compact model may converge more slowly. If a variant is still improving at 60, describe that limitation; any extension is an explicitly separate follow-on experiment, not a silent replacement.
- For newly trained runs, store full model state, configured channels, dropout, LR, seed, batch size, epoch budget, augmentation mode and probabilities, normalization statistics, selected epoch, selected validation metrics, class order, and software version in the checkpoint. The original clean trainer has CLI flags `--channels`, `--dropout`, `--lr`, and `--seed`, and its current new checkpoints explicitly store `channels` and `dropout`. The augmented trainer currently has no CLI flags and requires a targeted extension to cover A2/A3 and multiple seeds; this protocol is committed before those changes or runs. Older original checkpoints lack explicit configurable architecture metadata; interpret them using the verified original default, not a guessed nondefault configuration.
- Each new run writes to a **unique non-overwriting directory** under `cnn_outputs/`, with `checkpoint.pth` and `history.csv` plus reproducible configuration metadata. Original `cnn_rough_seed42.pth`, `cnn_rough_aug_seed42.pth`, and histories must remain unchanged.

## 4. Six-configuration hyperparameter comparison — 18 cells

All H configurations use **no training augmentation**. Run each at seeds 42, 1337 and 2026.

| ID | Channels | Params | Dropout | LR | Seed 42 |
|---|---|---:|---:|---:|---|
| H1 Original | 16/32/64/96 | 50,819 | 0.20 | 0.001 | Previously completed |
| H2 Medium | 12/24/48/72 | 28,803 | 0.20 | 0.001 | Previously completed |
| H3 Small | 8/16/32/48 | 12,995 | 0.20 | 0.001 | Previously completed |
| H4 Tiny | 4/8/16/24 | 3,395 (calculated; verify at runtime) | 0.20 | 0.001 | Pending |
| H5 No dropout | 16/32/64/96 | 50,819 | 0.00 | 0.001 | Pending |
| H6 Reduced LR | 16/32/64/96 | 50,819 | 0.20 | 0.0005 | Pending |

Do not add architecture depth, kernel-size, pooling, normalization, or large-grid search variants to this registered matrix. Preserve the original architecture as the canonical compression reference independently of whether a narrower architecture is attractive.

## 5. Four-way augmentation comparison — 12 cells, on original architecture

All A configurations use 16/32/64/96 channels (50,819 parameters), dropout 0.20, LR 0.001, and seeds 42, 1337 and 2026.

| ID | Training augmentation | Shift probability | Noise probability | Seed 42 |
|---|---|---:|---:|---|
| A1 None | None | 0 | 0 | Existing original clean checkpoint (same as H1) |
| A2 Shift | Independent ±500-sample circular roll | 0.5 | 0 | Pending |
| A3 Noise | Add 20 dB-SNR Gaussian white noise | 0 | 0.5 | Pending |
| A4 Both | Each transformation applied independently | 0.5 | 0.5 | Existing original augmented checkpoint |

Augment **training only**, before standardization; evaluate using unchanged validation windows for model selection. One training example is sampled per original window per epoch; do not silently substitute the different threefold dataset-expansion approach in shared `src/data.py`. Keep transformations' order, RNG behavior and probability semantics consistent with the original augmented run. If implementing configurable augmentation changes existing-seed behavior, document the discrepancy and assess whether the prior A4 seed-42 run is still a controlled match.

**Run-count reconciliation:** 18 H cells + 12 A cells − 3 shared H1/A1 clean-original cells = **27 distinct cells**. Four are verified complete (H1/A1 seed 42, H2 seed 42, H3 seed 42, A4 seed 42), leaving **23 new training runs**. Do not retrain shared cells merely because two tables reference them.

## 6. Validation-based comparison and model selection

- Test data is excluded from fitting, epoch selection, tuning, augmentation decisions, robustness screening, and selection of finalist candidates.
- For each configuration, report 3-seed mean and range/standard deviation of validation accuracy and validation cross-entropy, together with selected epochs. An explicitly registered clean-validation screening gate is **at least 26/27 correct windows for every seed**; report which configurations fail it. This threshold is a pragmatic screening rule, not a statistical confidence guarantee.
- For configurations clearing the gate, compare mean validation cross-entropy as the primary classification-sensitive score; report parameter count and stability, without claiming tiny validation-loss differences are significant. Keep both a performance-focused choice (lowest mean validation loss among eligible configurations) and a parameter-minimizing eligible alternative visible if those differ.
- Evaluate augmentation configurations on the same original architecture and seeds; consider clean-validation mean loss and the predetermined robustness results, not test results. The augmentation study is not a full architecture-by-augmentation factorial design; do not claim the best augmentation was proven on smaller widths.
- Before any final test scoring, **write down the selected checkpoint(s) or three-seed model set, reasons for selection, and the final analysis plan in a frozen selection record**. Retain the original baseline as the common compression reference; coordinate any other architecture with compression owners rather than assuming their loaders can reconstruct it.

## 7. Validation-only robustness protocol

Use the original **27 validation windows**, starting from raw signals. Construct deterministic corruption sets without touching the test split. Keep class labels and the clean training normalization statistics unchanged.

- Gaussian noise at SNR **20, 15, 10, 5 and 0 dB**. Use a documented fixed random seed (proposed `20261107`), the same per-window perturbations for every evaluated model, and consistent signal-RMS-relative SNR computation using the existing project's noise helper or a verified mathematically equivalent implementation.
- Circular shifts of **−500 and +500 samples**, using `np.roll` on raw windows, then fixed training normalization. These shifts exercise the range used in training augmentation and are exactly reproducible.
- Evaluate the three-seed models from all four augmentation conditions, and optionally the shortlisted hyperparameter variants if needed for the final comparison. Do not create additional training runs for this evaluation.
- Report per-SNR accuracy (and, where useful, validation cross-entropy), plus shifted accuracy, alongside the unperturbed baseline. Label this a **controlled validation robustness stress test**, not an estimate of unseen real-world operating conditions.
- Do not use these validation perturbations to repeatedly redesign the model or noise settings. Any material change to the corruption definition requires a documented protocol amendment before new evaluation.

## 8. Final test evaluation and benchmarking

Only after all validation screening, robustness analysis, reference-model decisions and selection records are frozen:

1. Perform **one final evaluation stage** on the provided 102-window test split for the fixed three-seed final model set(s). Each frozen model can be scored once in this stage; do not revisit architecture/hyperparameters using test results.
2. Report test accuracy for each seed, **mean and variability**, aggregate and per-class confusion matrices, precision, recall, macro-F1 (weighted-F1 optional). Include exactly which model and seed generated every result.
3. Report trainable parameter count, actual serialized checkpoint/model size in decimal MB, compression ratio when relevant, and CPU batch-size-one inference latency using the shared `src/metrics.py` methods and the team's `measurement-guide.md` protocol (10 warmup, 100 repeats, seed 42). Match CPU runtime/session and threading settings for comparative measurements; note preprocessing costs and hardware. Do not compare absolute latencies recorded on unrelated machines without qualification.
4. Integrate finalized rows into `results/experiments.csv` with the correct shared schema; keep separate validation-only tables and raw histories outside fields reserved for test accuracy. Coordinate CSV edits because multiple teammate branches have divergent versions.
5. Hand off the selected checkpoint, class order, training normalization statistics, loader instructions, architecture metadata, and a reproducible benchmark command to pruning/projection/quantization owners. Do not independently redo their assigned compression research.

## 9. Required output artifacts and acceptance criteria

- `docs/cnn_experiment_protocol.md` committed **before further training**, with explicit disclosure of prior completed runs.
- Configuration-specific saved checkpoints and histories, no overwrite of original files.
- A reproducible validation-run summary for H1–H6 × 3 and A1–A4 × 3, de-duplicating shared H1/A1 runs.
- Predefined robustness perturbations with fixed RNG and validation-only summary.
- Frozen validation selection record made before final test evaluation.
- Three-seed test report, confusion matrices, per-class metrics, model size and CPU latency.
- Coordinated team experiment-log update and concise limitations statement.

### Amendments and audit notes

**Local audit completed before protocol commit (2026-11-07):** `git status -sb` showed `feature/1d-cnn-optimization` with uncommitted edits to `src/cnn_1D/cnn_model.py` and `src/cnn_1D/cnn_train.py`, untracked medium/small run directories, and untracked Python cache folders. The local model signature was `(channels=(16, 32, 64, 96), dropout=0.2)`. `cnn_train.py` exposed `--channels`, `--dropout`, `--lr`, and `--seed`, and new checkpoints retained channels and dropout; `cnn_train_aug.py` had no CLI flags or configurable architecture checkpoint metadata. Both original and both new checkpoint histories contained 60 epochs; best epochs were 60 (original clean), 60 (original augmented), 57 (medium), 60 (small). This protocol is being registered **after those completed experiments but before additional training**. Post-commit deviations must be documented through a protocol amendment, not silently substituted.
