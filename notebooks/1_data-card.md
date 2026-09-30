# Data Card: Rolling Element Bearing Fault Diagnosis Dataset

## Source

- **Data owner:** Eric Bechhoefer ([eric@gpms-vt.com](mailto:eric@gpms-vt.com))
- **Original source:** [data-acoustics.com/measurements/bearing-faults/bearing-2](http://data-acoustics.com/measurements/bearing-faults/bearing-2)

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This means the data may be shared and adapted for non-commercial purposes, with attribution, and any derivative work must be shared under the same license.

## What the Data Is

Vibration acceleration signals from rolling element bearings, labeled into three classes:

- **Normal** — healthy bearing
- **OuterRaceFault** — fault on the outer race
- **InnerRaceFault** — fault on the inner race

Each sample is a single vibration signal segment of 5,000 points.

## Modifications from the Original Dataset

*(carried over from `data/README.md`)*

The original dataset contains 6 folders: baseline conditions, outer race faults, inner race faults, analyses, and real-world examples. For this project:

- Only the first 4 folders are used (baseline, outer race, inner race fault conditions) — the "analyses" and "real-world examples" folders were **dropped**.
- Signals were resampled to a consistent **48,828 Hz** sample rate.
- Signals were segmented into fixed-length windows of **5,000 samples**.
- The dataset was balanced across the 3 classes.
- Split into **train (75%)**, **validation (5%)**, and **test (20%)** sets.
- **Remaining preprocessing:** standardization has *not* been applied yet and should be done before model training.

## Known Limitation

Data comes from a **single test rig under controlled laboratory conditions**. Results may not generalize to real-world smart appliance sensors, different rig geometries, mounting conditions, or noise environments.

## Data Audit Results (this task)

Audit performed by loading each `.mat` file with `scipy.io.loadmat` and inspecting shapes, labels, dtypes, and signal values directly.

### 1. Split sizes / class balance

| Split | Expected shape | Actual shape | Expected per-class | Actual per-class | Status |
|-------|----------------|--------------|---------------------|-------------------|--------|
| train | (393, 5000) | (393, 5000) | 131 / 131 / 131 | 131 / 131 / 131 | ✅ Match |
| val   | (27, 5000)  | (27, 5000)  | 9 / 9 / 9 | 9 / 9 / 9 | ✅ Match |
| test  | (102, 5000) | (102, 5000) | 34 / 34 / 34 | 34 / 34 / 34 | ✅ Match |

No mismatches found — no flags needed.

### 2. Dtypes / NaN / Inf

| Split | Signal dtype | Label array dtype | NaN count | Inf count | Value range |
|-------|-------------|--------------------|-----------|-----------|-------------|
| train | float64 | object (string cells) | 0 | 0 | [-45.97, 39.32] |
| val   | float64 | object (string cells) | 0 | 0 | [-23.55, 23.73] |
| test  | float64 | object (string cells) | 0 | 0 | [-26.89, 27.50] |

- No missing (NaN) or infinite values in any split.
- All signal data is `float64`; labels load as an object array of Python strings (as expected from MATLAB cell arrays).
- Value ranges are broadly consistent across splits, though train has a wider max magnitude (~46 vs ~24–27), consistent with train simply containing more samples and thus a higher chance of an extreme outlier — not itself a red flag, but worth a histogram/outlier look during EDA.
- Each `.mat` file contains only the two expected variables (`{split}Data`, `{split}Labels}`) plus MATLAB's standard `__header__`/`__version__`/`__globals__` metadata — no stray variables.

### 3. Duplicate check

- **Within-split duplicates:** 0 duplicate signals found in train, val, or test (exact row-hash comparison via MD5 on raw signal bytes).
- **Cross-split duplicates:** 0 signals shared between train/val, train/test, or val/test.

No duplication or leakage-by-copy detected at the exact-match level. (Note: this checks for *identical* signals only — see open questions below regarding leakage from windowing overlap, which exact-match hashing would not catch.)

## Open Questions for Advisor

1. **Windowing vs. split order (leakage risk):** Was the 5,000-sample windowing/segmentation done *before* or *after* the train/val/test split? If a single longer recording was windowed first and *then* split, adjacent/overlapping windows from the same recording could end up in different splits, causing information leakage even though no exact-duplicate signals were found (near-duplicate, highly correlated windows wouldn't be caught by our exact-hash duplicate check).
2. **Dropped folders:** The original dataset had 6 folders (baseline, outer race fault, inner race fault, analyses, real-world examples); this project uses only the first 4. What exactly was in the "analyses" folder, and could the "real-world examples" folder be useful later (e.g., as an out-of-distribution test set to probe the single-test-rig limitation)?
3. **Resampling method:** What resampling method/filter was used to get to 48,828 Hz, and was it applied per-recording before or after segmentation? (Relevant for understanding whether resampling could introduce any train/val/test-correlated artifacts.)
4. **Outlier signals:** Train's max absolute value (~46) is notably higher than val/test (~24–27) — is this expected physical behavior (e.g., a specific fault severity present only in train) or worth flagging as a potential outlier/artifact?
