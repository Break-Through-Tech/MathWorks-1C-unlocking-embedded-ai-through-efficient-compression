# Amendment 1 — Micro CNN Capacity Experiment

Date: 2026-11-07
Original protocol: docs/cnn_experiment_protocol.md

## Reason for amendment

All 18 registered hyperparameter runs were completed
across six configurations and three seeds.

All 18 selected checkpoints achieved 100% accuracy
on the same 27 clean validation windows.

The smallest registered CNN had 3,395 parameters.
Consequently, the clean-validation capacity failure
point was not identified.

## Added experiment: H7 Micro CNN

- Channels: (2, 4, 8, 12)
- Expected parameters: 923
- Dropout: 0.20
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Training budget: 60 epochs
- Seeds: 42, 1337, 2026
- Augmentation: none
- Normalization: existing clean-training statistics
- Checkpoint selection: minimum clean-validation loss
- Test-set evaluation: prohibited during screening

## Interpretation

Compare H7 with the previously completed H1-H4
width configurations using validation accuracy,
validation cross-entropy and parameter count.

If H7 loses accuracy, report the observed capacity
trade-off. If it retains 100% accuracy across all
three seeds, stop reducing width and report that
the tested configurations did not identify the
clean-validation failure point.

Do not add more width configurations based on H7
results. Proceed with the registered augmentation,
robustness and benchmarking studies.

## Disclosure

This is a post-hoc amendment, not part of the
original preregistered six-configuration study.
It was motivated by the results of all 18
completed hyperparameter runs.

The original protocol and completed checkpoints
remain unchanged.
