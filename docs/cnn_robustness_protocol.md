
# CNN Validation Robustness Benchmark

Date: 2026-11-07

## Dataset

Use only the existing 27 validation windows.
Do not access the 102 test windows during robustness screening.

## Perturbations

### Gaussian Noise
- SNR: 20, 15, 10, 5, 0 dB
- Three deterministic noise realizations per SNR
- Random seed: 20261107
- Apply to raw signals before standardization

### Circular Time Shifts
- Offsets: -500, -250, +250, +500 samples
- Apply using np.roll to raw signals

## Evaluation

- Use identical perturbed windows across models.
- Keep validation labels and ordering unchanged.
- Use saved clean-training normalization statistics.
- No retraining or test-set evaluation.

## Models

Primary: Four augmentation conditions, three seeds each.

Secondary: Original, medium, small, tiny, and micro CNNs.

Record accuracy by perturbation severity and training seed.
Preserve individual model results.

## Coordination

Coordinate the benchmark with Farheen so both
1D and 2D models use identical raw perturbed windows.

## Limitations

Repeated perturbations are not independent recordings.
Synthetic robustness does not establish real-world reliability.
Recording-level independence remains unverified.
