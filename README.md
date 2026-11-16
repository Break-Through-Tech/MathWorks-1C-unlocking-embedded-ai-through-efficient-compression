# Smarter Devices, Smaller Models: Unlocking Embedded AI Through Efficient Compression

**Break Through Tech AI Studio × MathWorks | Fall 2026**  
**Task:** Three-class bearing-fault diagnosis from vibration signals, with accuracy–size–latency trade-off analysis for embedded AI.

This team project investigates how well machine-learning models can identify healthy and faulty rolling-element bearings while reducing the storage and computational demands of inference. We developed signal-processing features, classical and deep-learning classifiers, and multiple approaches to model compression. The objective is to inform embedded deployment decisions—not to claim that a model has already been deployed on a physical device.

> **Repository status (November 2026):** The dataset, exploratory notebooks, shared utilities, and SVM baseline are on `main`. The 1D CNN ([PR #23](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/23)), its optimization study ([PR #26](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/26)), the 2D CNN ([PR #24](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/24)), and projection ([PR #25](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/25)) are in open PRs; pruning and quantization are available on separate branches. Results below link to their actual locations and must not be interpreted as all being merged into `main`.

---

### 👥 **Team Members**

| Name | GitHub | Verified contributions |
|---|---|---|
| Gurbaj Singh | [@gurbaj8226](https://github.com/gurbaj8226) | Global-versus-per-window standardization study; PyTorch 1D CNN baseline; architecture/hyperparameter and augmentation experiments; robustness, frozen evaluation, and CPU benchmarking ([PRs #23 and #26](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/26)). |
| Sabina Ruzieva | [@sabinova](https://github.com/sabinova) | Waveform visualization and augmentation analysis; low-rank CNN projection experiments and write-up ([PR #25](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/25)); team task coordination. |
| Farheen Nashaat | [@farheen07456](https://github.com/farheen07456) | Frequency-domain and spectrogram analysis; 2D spectrogram CNN and synthetic-noise stress testing ([PR #24](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/24)). |
| Tule Achor | [@achoroj3](https://github.com/achoroj3) | Twelve signal-derived features and classical SVM baseline; recorded baseline evaluation and metrics. |
| Aigerim Zhadikbay | [@aigerim444](https://github.com/aigerim444) | Shared experiment-log/Pareto-plot scaffolding; structured and unstructured pruning, sparse model serialization, and pruning evaluation (`aigerim/pruning`). |
| Venya Goyal | [@Venya07](https://github.com/Venya07) | Shared data-loading and evaluation utilities; post-training static INT8 quantization and artifact measurements (`quantization`). |
| Leo | [@leowuniv](https://github.com/leowuniv) | Dataset audit and data-card work; shared measurement guide and environment/dependency updates. |

---

## 🎯 **Project Highlights**

- Built a **12-feature SVM baseline** (98.04% accuracy, 100/102 test windows) and developed **raw-waveform 1D** and **spectrogram-based 2D CNNs**.
- Evaluated five dense 1D CNN widths, from **50,819** down to **923 trainable parameters**, plus learning-rate, dropout, and four training-augmentation conditions across a documented multi-seed study.
- The selected **923-parameter micro CNN** matched the original CNN's **102/102 test accuracy** on the provided split; its inference-package size was **0.0136 MB versus 0.2148 MB**, with a **42.2% lower mean CPU latency in one matched Google Colab session**.
- Explored **unstructured/structured pruning**, **post-training static INT8 quantization**, and **low-rank projection**, recording how storage and accuracy respond to compression. These methods were evaluated in separate branches and environments; their reported CPU times are not yet a single fair team-wide ranking.
- Preserved reproducible datasets, scripts, notebooks, metrics, and supporting documentation, while explicitly tracking limitations of the controlled laboratory dataset.

---

## 👩🏽‍💻 **Setup and Installation**

### 1. Clone and install

Use Python 3 with `venv` and the packages in [`requirements.txt`](requirements.txt): NumPy, SciPy, pandas, scikit-learn, Matplotlib, and PyTorch.

```bash
git clone https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression.git
cd MathWorks-1C-unlocking-embedded-ai-through-efficient-compression
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install Jupyter separately if you want to run the notebooks (`python -m pip install jupyter`). PyTorch builds and INT8 backends depend on your platform. Dependencies are not version-pinned, so exact environment recreation may require the versions recorded in the linked benchmark reports.

### 2. Load data and reproduce the classical baseline (`main`)

The provided [`data/`](data/) directory contains `train.mat`, `val.mat`, and `test.mat` plus derived CSV feature files; no external download is required for the supplied project splits. Run from the repository root:

```bash
python -c "from src.data import load_all_splits; print({k: v[0].shape for k, v in load_all_splits().items()})"
```

To regenerate the feature CSVs and train/evaluate the SVM, run from `src/` because these scripts use relative paths:

```bash
cd src
python build_features.py
python build_model.py
cd ..
```

The SVM scripts write a model and evaluation log under `src/output/`. See [`src/build_features.py`](src/build_features.py), [`src/build_model.py`](src/build_model.py), and [`src/output/log.txt`](src/output/log.txt).

### 3. Reproduce the CNN and compression studies (feature branches)

The CNN code is **not currently on `main`**. In a fresh clone, fetch and check out the optimization branch before running its commands:

```bash
git fetch origin
git switch --track origin/feature/1d-cnn-optimization
python -m src.cnn_1D.cnn_smoke_test
python -m src.cnn_1D.cnn_final_team_benchmark
```

The final benchmark reloads packaged models and measures them in the **current** CPU environment; expect latency to vary by machine. The [CNN technical handoff](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/docs/cnn_optimization_handoff.md) documents training, model loading, checkpoint provenance, perturbation tests, and additional reproduction commands. For pruning, quantization, projection, or the 2D CNN, check out the corresponding branch and follow its documentation linked below; do not run branch-specific module commands on `main`.

---

## 🏗️ **Project Overview**

This project is being developed as a part of the **Break Through Tech AI Studio fellowship**, in collaboration with **MathWorks**, whose tools support engineering, technical computing, and model-based design. The [MathWorks challenge brief](Challenge-Project-Overview.md) asks for a deep-learning fault-diagnosis model, analysis of **accuracy, inference speed, and memory/storage requirements**, and at least **two compression techniques** with documented trade-offs.

Bearings are components of motors and rotating equipment. Detecting inner- and outer-race faults from vibration sensors can support condition monitoring and preventive maintenance. For embedded applications, a highly accurate classifier is not enough: model storage, compute cost, and latency also matter. Our work compares modeling and compression strategies against those constraints using an offline experimental dataset.

---

## 📊 **Data Exploration**

**Dataset:** [Rolling Element Bearing Fault Diagnosis](data/README.md), attributed to Eric Bechhoefer and distributed for this challenge as MATLAB `.mat` files. Signals were resampled to **48,828 Hz**, divided into **5,000-sample windows**, balanced across three labels, and provided as fixed splits:

| Split | Windows | Normal | InnerRaceFault | OuterRaceFault |
|---|---:|---:|---:|---:|
| Train | 393 | 131 | 131 | 131 |
| Validation | 27 | 9 | 9 | 9 |
| Test | 102 | 34 | 34 | 34 |
| **Total** | **522** | **174** | **174** | **174** |

The [data card](notebooks/1_data-card.md) verifies shapes, labels, and the absence of NaN/Inf values or exact duplicate windows. **It does not establish that windows from different splits came from independent source recordings.**

**Exploration and preprocessing.** The team examined waveforms, fault impulses, RMS, kurtosis, amplitude variability, FFT/band energy, envelope spectra, and spectrograms. An [augmentation study](notebooks/2_SignalVisualization_Augmentation.ipynb) supported circular shifts and moderate Gaussian noise, while avoiding amplitude scaling that could obscure relevant class-level magnitude differences. The [standardization study](notebooks/05_standardization_study.ipynb) found that **global z-score normalization fitted on the clean training split** preserves between-window amplitude differences better than independently standardizing each window. The same training statistics are applied to validation and test data. The supplied train/validation/test split is not reshuffled.

**Exploration artifacts:** [data preparation](notebooks/1_data_clean_updated.ipynb) · [signal visualization and augmentation](notebooks/2_SignalVisualization_Augmentation.ipynb) · [frequency-domain analysis](notebooks/3_Frequency_Domain_Analysis.ipynb) · [standardization analysis](notebooks/05_standardization_study.ipynb) · [dataset details](data/README.md).

---

## 🧠 **Model Development**

### Classical and deep-learning models

- **SVM baseline (`main`):** Twelve features consisting of RMS, kurtosis, and standard deviation calculated on the time signal, Fourier magnitudes, spectrogram, and power spectral density. An SVM was fitted and evaluated using the supplied splits; its reported inference timing covers prediction on **precomputed features**, not feature extraction from raw sensor data.
- **Original raw-signal 1D CNN ([PR #23](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/23)):** Four `Conv1d → BatchNorm → ReLU → MaxPool` blocks (16/32/64/96 channels), adaptive average pooling, dropout, and a three-class output head; **50,819 trainable parameters**. It is the common **dense reference for compression**, not the optimized micro model.
- **Optimized dense 1D CNN ([PR #26](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/26)):** Tested original, medium (**28,803** parameters), small (**12,995**), tiny (**3,395**), and micro (**923**) channel widths. The micro model is a **separately trained narrower dense architecture**, *not* a pruned or quantized version of the original.
- **2D spectrogram CNN ([PR #24](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/24)):** A small `TinySpecCNN` trained on **129 × 22 log-dB spectrograms** generated from the vibration windows, with separate noise stress testing.

### Tuning, augmentation, and selection

The 1D CNN study documents **30 distinct training runs**, including width, learning-rate, dropout, and augmentation comparisons using seeds **42, 1337, and 2026**. The common training setup used Adam, cross-entropy loss, batch size **32**, typically learning rate **0.001**, and **60 epochs**; checkpoints were selected by lowest clean-validation loss. Four augmentation conditions were compared on the original CNN: **none, circular shift (±500 samples), Gaussian noise (20 dB SNR), and both**, with each transform independently applied with 50% probability where enabled. Augmentation was training-only.

All selected training-run checkpoints reached **27/27 clean-validation accuracy**, so that small validation set could not meaningfully rank models by accuracy. A frozen evaluation of **18 selected checkpoints** subsequently found **102/102 test accuracy** and **macro-F1 of 1.00** for each. These scores are observations on the same limited dataset, **not evidence of perfect generalization**. Severe synthetic-noise robustness varied by seed: for example, the micro CNN's three-seed mean at **0 dB SNR** was **87.24%**, but its packaged seed-1337 checkpoint scored **69.14%**. No consistent advantage from the tested training augmentations was established.

### Compression investigations

| Technique | Implemented approach | Status and important distinction |
|---|---|---|
| **Unstructured / structured pruning** | Remove individual low-magnitude weights or entire low-norm filters, followed by fine-tuning; save surviving weights in compressed sparse archives. | [`aigerim/pruning`](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/tree/aigerim/pruning). Sparse files save storage, but models are reconstructed as dense networks for inference; **no reliable CPU speedup** was demonstrated. |
| **Static INT8 quantization** | Fuse convolution, batch normalization, and ReLU; calibrate on training windows; convert to 8-bit inference weights/activations. | [`quantization`](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/tree/quantization). Backend-specific results; both state-dictionary and TorchScript files were saved. |
| **Low-rank projection** | Apply truncated SVD to replace the original CNN's larger convolution layers with two lower-rank layers, with optional fine-tuning. | [PR #25](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/pull/25). Reported results are **validation-only**; final test-set comparisons are not yet documented. |

**Combined compression** (such as pruning + quantization or projection + quantization) and a final team-wide deployment-model decision remain outstanding; no integrated result is claimed here.

---

## 📈 **Results & Key Findings**

The table combines **verified but branch-specific** reports. **Sizes refer to the stated saved file format** (decimal MB), and CPU measurements come from different machines/sessions unless specifically noted. Consequently, this is an **evidence inventory, not a valid cross-method speed ranking**.

| Model / method | Accuracy on supplied test (102) | Saved size | Mean CPU latency | Artifact / measurement context |
|---|---:|---:|---:|---|
| 12-feature SVM (`main`) | **98.04%** | **0.00496 MB** | **0.1753 ms** | Pickle; inference on precomputed features, feature extraction excluded. |
| Original dense 1D CNN (reference) | **100%** | **0.21479 MB** | **1.2507 ms** | Inference package; matched Colab CPU session with micro CNN. |
| Micro dense 1D CNN (923 parameters) | **100%** | **0.01364 MB** | **0.7230 ms** | Inference package; same Colab session as original. |
| 2D spectrogram CNN | **100%** | **0.03853 MB** | **1.0828 ms** | Separate report; timing includes spectrogram computation. |
| Unstructured pruning, 90% | **100%** | **0.02624 MB** | **1.3740 ms** | Compressed sparse `.npz`; separate Colab CPU pruning session. |
| Unstructured pruning, 97% | **100%** | **0.01203 MB** | **1.4491 ms** | Sparse `.npz`; close to the observed accuracy break point. |
| Static INT8 CNN | **100%** | **0.05927 MB** | **0.5754 ms** | INT8 state dictionary; Apple ARM64 CPU, prestandardized input; TorchScript file is **0.09595 MB**. |
| Projected CNN (2% rank kept, fine-tuned) | **Not evaluated on test** | **0.03826 MB** | **~2.0 ms** | Dense `.pth`; **100% validation accuracy (27/27)**, Windows laptop CPU. |

**Main findings:**

1. **Dense width reduction:** Original and micro 1D CNN packages achieved the same observed test accuracy. In one matched Colab session, micro used **93.6% less serialized storage** and had **42.2% lower mean CPU latency**. This comparison is specific to those packages and that session.
2. **Pruning:** At **90% unstructured pruning**, the sparse archive was approximately **8.3× smaller** than its same-session dense reference with no observed accuracy loss; the **97%** variant was approximately **18× smaller**, but was closer to failure. Structured pruning degraded substantially at aggressive removal levels. Storing zeros compactly did **not** make dense execution faster.
3. **Quantization:** In the matched Apple ARM64 run, INT8 and FP32 both achieved **100% observed test accuracy**; state-dictionary storage decreased from **0.21527 to 0.05927 MB** (about **3.6×**), and mean timing changed from **0.6091 to 0.5754 ms**. These timings should not be compared directly with Colab results.
4. **Projection:** At its most aggressive tested level, fine-tuned projection reduced the `.pth` file from approximately **0.213 to 0.038 MB** (about **5.6×**) while preserving **27/27 validation accuracy**. This is **not** a verified test-set result or a demonstrated inference speedup.

**Evaluation limits:** The provided data originates from **one controlled test rig** and contains only **27 validation** and **102 test** windows. Exact-window duplicate checks passed, but **recording/session independence and windowing-before-splitting provenance remain unverified**. Multiple CNN candidates were assessed on the same test set, which had also been inspected elsewhere in the team workflow. Perfect scores therefore do not prove reliability on independent recordings or unseen devices. Synthetic-noise trials are not real-world deployment validation, and **none of these CPU benchmarks measures an embedded processor**. A single-session, preprocessing-consistent benchmark across *all* team models is still needed.

---

## 📚 **Technical Documentation & Artifacts**

| Resource | Location |
|---|---|
| Challenge objectives and milestones | [`Challenge-Project-Overview.md`](Challenge-Project-Overview.md) |
| Dataset provenance, audit, and license | [`data/README.md`](data/README.md) · [`notebooks/1_data-card.md`](notebooks/1_data-card.md) |
| All exploration notebooks | [`notebooks/`](notebooks/) |
| Shared measurement protocol | [`measurement-guide.md`](measurement-guide.md) · [`src/metrics.py`](src/metrics.py) |
| Main-branch baseline results and plotting script | [`results/experiments.csv`](results/experiments.csv) · [`results/pareto_plot.py`](results/pareto_plot.py) |
| CNN experiment protocol, model selection, test results, and reproducibility | [CNN optimization handoff](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/docs/cnn_optimization_handoff.md) · [final test evaluation](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/cnn_outputs/final_eval/frozen_test_results_v1.json) · [model manifest](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/cnn_outputs/final_models/manifest.json) |
| Matched CPU benchmark (original versus micro) | [Colab CPU benchmark JSON](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/cnn_outputs/benchmarks/final_colab_cpu_v1.json) |
| CNN augmentation and robustness measurements | [augmentation CPU benchmark](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/feature/1d-cnn-optimization/cnn_outputs/benchmarks/augmentation_colab_cpu_v1.json) · [robustness CSVs](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/tree/feature/1d-cnn-optimization/cnn_outputs/robustness) |
| Pruning report, chart, and reproduction instructions | [pruning write-up](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/aigerim/pruning/results/pruning_writeup.md) · [pruning accuracy chart](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/aigerim/pruning/results/pruning_accuracy_vs_level.png) |
| Quantization notebook, saved models, and results | [quantization notebook](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/quantization/notebooks/06_quantization_colab.ipynb) · [INT8 results JSON](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/quantization/cnn_outputs/quantized/static_int8_histogram_qnnpack/results.json) |
| Projection methods, sweep data, and chart | [projection write-up](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/projection/results/projection_writeup.md) · [projection chart](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/projection/results/projection_sweep.png) |
| 2D CNN methods and synthetic-noise visualization | [2D CNN documentation](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/2d-cnn-spectrogram/src/cnn_2D/README.md) · [noise chart](https://github.com/Break-Through-Tech/MathWorks-1C-unlocking-embedded-ai-through-efficient-compression/blob/2d-cnn-spectrogram/cnn_outputs/noise_stress_test_chart.png) |

---

## 🚀 **Next Steps**

- **Integrate and validate:** Review/merge outstanding PRs and compression branches, reconcile overlapping experiment IDs, and verify a reproducible installation and run path from the consolidated `main` branch.
- **Finish comparable evaluation:** Retest the selected candidate models and any combined-compression variants on the **same CPU host/session**, with consistent timing boundaries, saved-file definitions, and feature-extraction/preprocessing accounting; then generate final size/latency Pareto charts.
- **Complete missing experiments:** Final test evaluation of projected candidates; combined compression and cross-method robustness comparisons; final model choice based on accuracy, footprint, latency, and robustness together.
- **Strengthen external validity:** Confirm original recording identities and segmentation order; evaluate recording-disjoint and cross-rig datasets if available; test under realistic noise and eventually on target embedded hardware.
- **Improve reproducibility:** Pin tested dependency versions, document hardware/backend requirements, and retain manifests/checkpoints for final reproducible model comparisons.

---

## 📝 **License**

The **dataset** is attributed to Eric Bechhoefer and documented as **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**; see [`data/README.md`](data/README.md) for provenance and reuse conditions. **No separate repository-wide source-code license is currently declared.** Do not assume that the dataset license automatically licenses the team's code; confirm any code-license decision with the team and Challenge Advisor before adding one.

---

## 📄 **References**

- [MathWorks AI Studio Challenge Overview](Challenge-Project-Overview.md).
- [Rolling Element Bearing Fault Diagnosis dataset provenance and license](data/README.md).
- [MathWorks: Rolling Element Bearing Fault Diagnosis Using Deep Learning](https://www.mathworks.com/help/predmaint/ug/rolling-element-bearing-fault-diagnosis-using-deep-learning.html).
- [PyTorch: Pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).
- [PyTorch: Quantization documentation](https://docs.pytorch.org/docs/stable/quantization.html).

---

## 🙏 **Acknowledgements**

We thank **MathWorks** for sponsoring the AI Studio challenge and advising the team; **Shanmukh Srinivas**, our MathWorks Challenge Advisor; **Rashidah Carr**, our Break Through Tech AI Studio Coach; and the **Break Through Tech AI** program for project guidance and support. We also acknowledge **Eric Bechhoefer** for the underlying bearing-fault data used in this study.
