# How We Measure Results (Team Guide)

Goal: every model (baseline, pruned, quantized, projected, combined) is measured the
same way, so rows in `results/experiments.csv` can be compared fairly.

## 1. Use only the shared functions in `src/metrics.py`

| What | Function | Notes |
|------|----------|-------|
| Accuracy | `metrics.accuracy(y_true, y_pred)` | Returns a fraction in [0, 1]. |
| Model size | `metrics.measure_model_size_mb(model=..., kind="sklearn" or "pytorch" or "keras")` or `metrics.measure_model_size_mb(path="...")` | Decimal MB (bytes / 1,000,000), measured on disk. |
| Speed | `metrics.measure_inference_latency_ms(predict_fn, X)` | Returns a dict. **Log `mean_ms`.** |

Do not write your own accuracy, size, or timing code.

## 2. Rules for a fair measurement

- **Accuracy is on the test split only** (102 clips, so one clip is about 0.98%).
  Choose hyperparameters on val (27 clips), never on test.
- **Size is the file you would actually ship.** For a compressed model, save the
  compressed version and use `path=` on that file. Do not measure the original model
  and label it compressed. Note: pruning only zeroes weights, so a dense file may not
  get smaller unless it is saved in a sparse or compressed form. Say in `notes` how
  the file was saved.
- **Speed is batch size 1, one sample at a time.** Keep the defaults
  (`n_warmup=10`, `n_repeats=100`, `seed=42`). Do not change them.
- **Same type of computer for every timing.** Use a Google Colab **CPU** runtime
  (Runtime > Change runtime type > CPU, no GPU/TPU). Colab hardware can differ between
  sessions, so when you compare a compressed model to its original, **re-time both in
  the same session**.
- **State what the model's input is** in `notes` (extracted features vs. raw
  5,000-sample window). Timing a classical model on 12 precomputed features is not the
  same job as timing a deep model on a raw window. **Team decision needed:** whether
  latency for feature-based models should also include feature extraction time.
- **Standardization:** global z-score using train statistics (`data.standardize`).

## 3. Filling in `results/experiments.csv`

| Column | Who fills it | What to put |
|--------|--------------|-------------|
| `experiment_id` | Script | Automatic (next number). |
| `date` | You | Same format as existing rows, e.g. `10/21/2026`. |
| `author` | You | Your full name. |
| `model` | You | e.g. `SVM`, `1D CNN`. |
| `input_repr` | You | The feature list, or `raw 5000-sample window`. |
| `standardization` | You | `Global z-score` (team choice) or `None`. |
| `augmentation` | You | `None`, or what you used (time shift, noise). Train split only. |
| `compression` | You | `None`, `Pruning`, `Quantization`, `Projection`, or combined like `Pruning + Quantization`. Use these exact words: the plot groups by this column. |
| `accuracy_test` | Script | From `log.txt`. |
| `model_size_mb` | Script | From `log.txt`. |
| `latency_ms_per_sample` | Script | `Mean Latency` (`mean_ms`) from `log.txt`. |
| `notes` | You | Hyperparameters, how the file was saved, input type, anything odd. |

### Steps (run from the `src/` folder; paths are relative)

1. Produce `output/log.txt`. `build_model.py` does this for the baseline. For your own
   model, write a file in the same format. **The first three lines must be, in order:**
   ```
   Accuracy Score: <number>
   File size (MB): <number>
   Mean Latency: <number>
   ```
2. Write `output/experiments.txt` with these fields **in this order**, one per line,
   with `: ` after the name (`Notes` may span several lines):
   ```
   Author: ...
   Date: ...
   Model: ...
   Features: ...
   Standardization: ...
   Augmentation: ...
   Compression: ...
   Notes: ...
   ```
3. Check the parse: `python update_experiments.py test`
4. Save the row: `python update_experiments.py save`
   (`update` fixes the last row, `remove` deletes it.)
5. Commit `results/experiments.csv`.

## 4. Final charts

From the repo root: `python results/pareto_plot.py`
Writes `results/pareto_accuracy_vs_size.png` and `results/pareto_accuracy_vs_latency.png`.
