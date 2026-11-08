# CNN Final Evaluation Selection Freeze

Date: 2026-11-07
Branch: feature/1d-cnn-optimization

## Status
All 30 distinct training runs completed.
Hyperparameter and augmentation studies completed.
Validation robustness and dense CPU benchmarking completed.

## Frozen model groups
Evaluate all three seeds (42, 1337, 2026) for:

1. Original CNN, no augmentation
2. Original CNN, shift-only augmentation
3. Original CNN, noise-only augmentation
4. Original CNN, combined augmentation
5. Original CNN, dropout 0.00, no augmentation
6. Micro CNN (2/4/8/12), no augmentation

Total: 18 distinct checkpoints.

## Selection rationale
Original clean: canonical compression reference.
Four augmentation modes: original milestone comparison.
Dropout 0.00: lowest mean validation loss in H1-H6.
Micro: smallest tested dense architecture; 923 parameters,
100% clean-validation accuracy across three seeds,
and strongest observed severe-noise robustness among widths.

## Final evaluation policy
Use the existing 102-window test split once in a single
final evaluation stage, after this freeze is committed.

Do not tune, retrain, or change candidates using test results.

Report per-seed test accuracy, three-seed mean and variability,
confusion matrices, precision, recall and macro-F1.

Report final efficiency measurements separately.
Preserve original training checkpoints and histories.

## Limitations
Validation uses only 27 original windows.
Perturbed validation windows are not independent recordings.
Recording-level split independence remains unverified.
Some teammates have previously viewed test results;
this is not a completely blind team test.
