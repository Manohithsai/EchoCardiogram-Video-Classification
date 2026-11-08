# Echo-EF

A video-based cardiac function research project on EchoNet-Dynamic. Echo-EF estimates left ventricular ejection fraction (EF), end-diastolic volume (EDV) and end-systolic volume (ESV) from apical four-chamber echocardiogram videos. It compares models, checks failure patterns and keeps reliability experiments separate from the main prediction.

- Main application: R3D-18 regression, with three reporting groups derived from predicted EF.
- Supporting research: four-class R3D augmentation experiments, a matched Video Swin control/MixUp pair, uncertainty methods, calibration and model-influence checks.

Research prototype. Not a clinically validated diagnostic system. Not for medical diagnosis, treatment decisions or patient triage.

This file covers setup and run, the application, verified results, run history, data requirements and limitations.

**Contents:** [Run it yourself](#run-it-yourself-local-demo-website) - [Step-by-step on macOS](#step-by-step-on-macos) - [Step-by-step on Windows](#step-by-step-on-windows) - [Using the website](#using-the-website) - [Repository structure](#repository-structure) - [Verified results](#verified-results) - [Run history and transparency](#run-history-and-transparency) - [Data and models](#data-and-models) - [Known limitations](#known-limitations) - [Project rules](#project-rules)

## Run it yourself (local demo website)

**Packaging status, 8 November 2026:** the current application has been run and accepted on the project's Windows laptop. The public repository still contains an older version of the project. The steps below describe the accepted local application; a fresh clone alone is not yet a complete, tested demo package. Repository cleanup and release packaging remain separate work.

Unlike an upload-any-video demo, this version selects **local EchoNet-Dynamic TEST samples only**. You need an authorized local dataset copy and the original compatible model files. There is no arbitrary video upload in the accepted interface.

What you need:

- The current project folder, including `app/app.py`, its `src/` modules and `configs/base.yaml`.
- A compatible Python environment with PyTorch, torchvision and the application dependencies. Preserve the working laptop environment if you already have it. The repository does not yet provide a fully pinned, clean-install environment for the final app.
- The original baseline checkpoint and the local files listed under [Data and models](#data-and-models).
- An authorized EchoNet-Dynamic download. Neither videos nor per-video prediction archives are bundled for download here.
- A browser. The server binds to `127.0.0.1`, uses port `7860` by default and does not create a public Gradio share link.

Quick start for the already configured Windows project, from its root folder:

```powershell
python .\app\app.py
```

Open <http://localhost:7860>. Leave the terminal open while using the app. Stop it with `Ctrl+C`.

The application performs inference. Starting it does not train models, fit calibration, run a dataset-wide evaluation or download weights. It chooses CUDA if available; otherwise it uses CPU. No minimum RAM, disk requirement or inference-time guarantee is claimed because these have not been established for a portable release.

### Getting the code

The existing source repository is <https://github.com/Manohithsai/EchoCardiogram-Video-Classification>.

You can use **Code > Download ZIP**, or clone it:

```bash
git clone https://github.com/Manohithsai/EchoCardiogram-Video-Classification.git
cd EchoCardiogram-Video-Classification
```

**Do not mistake this for a finished installer:** at the time of this review, the public copy does not contain the accepted laptop app and all its current supporting files. Use the complete local project for the instructions below. Do not overwrite the working local folder with the older public copy.

### Model download

**No verified public checkpoint download or automatic model downloader is available.** The model filenames below describe local dependencies, not downloadable repository content. Keep the original compatible files on the local machine. Do not download a similarly named checkpoint and assume it reproduces these results, and do not retrain simply to launch the app.

The public repository, a ZIP of its source code and this README do not include the trained models. A clean public demo cannot be promised until a reviewed distribution plan and reproducible environment are prepared.

## Step-by-step on macOS

**Not yet tested end-to-end on a Mac.** These are the corresponding Terminal commands for a complete local project, not a claim of verified macOS support. No Apple Silicon speed, memory or MPS acceleration result is available. The current app uses the CPU fallback when CUDA is unavailable; it does not select Apple's MPS backend.

1. Obtain the complete current project folder and the required local artifacts. Download EchoNet-Dynamic through its official access process, accepting its terms yourself. Do not use the older public clone as if it already includes the accepted app.
2. Open Terminal in the project root, the folder containing `src`, `configs` and `app`. Check that `python3 --version` points to the Python environment you intend to use. A specific Mac Python/PyTorch/torchvision version combination has not been verified for this app.
3. If creating a new environment, use:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

4. Install a compatible PyTorch and torchvision pair using the official selector at <https://pytorch.org/get-started/locally/>. Then install the repository dependencies and the app-specific packages:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install gradio==6.31.0 matplotlib
   ```

   The current app explicitly checks for `gradio==6.31.0`. The broad repository requirements alone are not a reproducible environment lock. A dependency installation failure is a setup gap to fix, not a reason to change model weights or retrain.

5. Set the local dataset paths in `configs/base.yaml` as described under [Data and models](#data-and-models). Place the original baseline checkpoint at its expected project-relative location. The local app reads raw TEST videos directly; building the training cache is not required just to use this interface.
6. Start from the project root:

   ```bash
   python app/app.py
   ```

7. Open <http://localhost:7860>. Select a local TEST sample, analyze it and check the status message. Stop with `Ctrl+C`.

For later starts, activate the same environment and run `python app/app.py` again. If inference or explanation generation fails on a Mac, retain the error and report the exact environment. Do not describe a successful package installation as a verified application run.

## Step-by-step on Windows

The accepted app was run on the project's existing Windows laptop. A separate clean-machine installation has not been verified. If that working environment is already present, do not reinstall PyTorch or change CUDA just to follow these instructions.

1. Open PowerShell in the complete project root. It must contain `app\app.py`, `src\` and `configs\base.yaml`.
2. Use the environment that already runs the project. For a new setup only, create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

   If PowerShell blocks activation, you can use `.\.venv\Scripts\python.exe` in place of `python` for the installation and launch commands below. Changing the machine-wide execution policy is not necessary.

3. For a new environment only, install a compatible PyTorch/torchvision pair for CPU or the installed CUDA setup using <https://pytorch.org/get-started/locally/>. Do not guess a CUDA wheel version. Then:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install gradio==6.31.0 matplotlib
   ```

   This is an installation recipe, not a fully pinned, independently tested clean-install release. Keep the existing working torch/CUDA installation unchanged.

4. Check the dataset entries in `configs/base.yaml` and the original model files listed below. Nothing is automatically downloaded by the app.
5. Run from the project root:

   ```powershell
   python .\app\app.py
   ```

6. Open <http://localhost:7860>. The browser may open automatically. Keep PowerShell running; press `Ctrl+C` there to stop the server.

If port 7860 is occupied, stop the old app process, or use a different port explicitly:

```powershell
python .\app\app.py --port 7861
```

Then open <http://localhost:7861>. Starting another copy does not fix missing data or checkpoints.

## Using the website

The interface has four tabs: **Analyze**, **Model influence**, **Model results** and **Research**.

### Analyze

1. Select one of the available local TEST samples. The list comes from `FileList.csv` and existing original video files, sorted by filename rather than model result. Missing or ambiguous entries are not usable samples.
2. Run the analysis. The main predictor stays the original R3D-18 baseline. The app uses one centered 32-frame clip, stride 2, resized to 112 x 112; short clips are loop-padded. Inference uses the verified FP32 path.
3. Read the estimated EF and its derived reporting group:

   | Reporting group | Predicted EF |
   |---|---|
   | Reduced | Below 40 |
   | Mildly reduced | 40 to below 50 |
   | Normal | 50 or above |

   These are project EF cutoffs, not diagnoses. "Normal" describes this EF group only, not the entire scan or the patient's health. The underlying model also produces EDV and ESV.
4. If an experimental group-correctness percentage is available, interpret it as an estimate that the **derived group** matches the recorded group. It is not the probability that the exact EF is correct, a calibrated four-class head probability or a medical safety score. The app withholds it when its identity or fitted-domain checks fail.
5. Use the recorded-EF comparison only after prediction. The app checks the selected file against the configured original video and checks the label source. Recorded EF is a reference, not an input to the predictor. A group match can coexist with a large numeric error.

The accepted laptop example showed EF `59.9`, the Normal group and a `94%` experimental group-correctness estimate. The reference panel showed recorded EF `68.71%`, prediction `59.87%` and displayed absolute error `8.83` EF points. These are one example, not a performance benchmark. Subtracting the rounded labels gives `8.84`; the app computes error before separately formatting the displayed values. The captured display is retained rather than inventing extra precision.

### Model influence

Generate the explanation after analyzing a sample. Compare the original sampled frame with the positive EF attribution overlay, inspect the frame slider and the temporal view. The explanation must match the prediction's input and checkpoint.

The map is enlarged from a coarse feature grid. Warm colors indicate positive influence toward a higher EF estimate, not disease, verified anatomy or a causal explanation. An empty map or a failed explanation is not evidence that the heart is normal.

### Model results and Research

Model results are saved evaluations with their source-specific caveats. Clicking through the interface does not rerun the historical TEST benchmark or select a new main model.

Research keeps the residual-error model separate. It uses its **own** mean checkpoint, error head and calibration, so its range must not be attached to the main baseline EF. The four-class R3D models and the newer matched Swin pair also remain supporting experiments, not silent replacements for the main app.

### When something is unavailable

- **No TEST samples:** check `data.raw_video_dir` and `data.file_list_csv`, the CSV columns and the original video files. The app does not fetch the dataset.
- **Missing checkpoint:** restore the exact original local file. Do not rename a different model to satisfy the filename.
- **Estimate unavailable:** the correctness artifact may be missing, mismatched or outside its fitted domain. Do not bypass that warning.
- **Residual experiment unavailable:** check its complete mean/head/calibration set. The main predictor can remain separate.
- **GPU memory error:** close other GPU work and retry. Do not treat a failed request as a prediction.
- **Wrong working folder:** open the project root before running `python app/app.py`.

## Repository structure

The core source folders below are established project components. The `app/` entry and newer local experiment artifacts belong to the accepted local project; their presence in a future public release must be checked after cleanup. This is not a claim that every item already exists in the public branch.

```text
project-root/
├── app/
│   └── app.py                   # accepted local Gradio entry point
├── src/
│   ├── data/                    # dataset and clip handling
│   ├── models/                  # R3D, frame and Swin architectures
│   ├── train/                   # training implementations
│   ├── eval/                    # evaluation implementations
│   ├── infer/                   # video inference and residual-pair inference
│   └── xai/                     # explanation code
├── scripts/                     # cache builder and other utilities
├── configs/
│   └── base.yaml                # local data paths and experiment settings
├── tests/                       # tests; other test_*.py files also exist
├── docs/                        # experiment registration and launch records
├── analysis_archive_2026-10-29/  # older exploratory analyses
├── requirements.txt             # broad project dependencies, not a final lockfile
└── README.md
```

Local runtime files are documented below but are not source-code downloads. Keep videos, checkpoints, caches, per-video IDs, prediction arrays and private working files out of any new publication. Only reviewed aggregate metrics and safe documentation should accompany the cleaned source release.

## Verified results

These numbers come from completed runs and saved evaluations. **VAL and TEST were both used during development.** "Historical TEST" is a split label, not a claim of an untouched final test or external validation. TRAIN has 7,465 videos, VAL 1,288 and TEST 1,277, for 10,030 total.

MAE and RMSE are EF percentage points, not percent accuracy. True R² means `1 - SSE/SST`. F1 is on a 0 to 1 scale. Rounded results retain the source evaluation's identity.

### Main R3D-18 baseline: EF regression

| Saved evaluation | Videos | MAE | RMSE | True R² |
|---|---:|---:|---:|---:|
| VAL, FP32, one centered clip | 1,288 | 5.4671 | 7.3237 | 0.6458 |
| Historical TEST, checkpoint recheck | 1,277 | 5.3793 | 7.3455 | 0.6390 |
| Later TEST evaluator, separate version | 1,277 | 5.3796 | 7.3457 | 0.6390 |

The main baseline has about **5.38 EF-point average historical TEST error**. That is not "94.62% accuracy" and is not a per-video error guarantee. The old field named `r2` in an earlier historical file was squared Pearson correlation, not true R².

The main checkpoint is the original epoch-24 baseline. Its SHA-256 is:

```text
7ea9c83359e3918019a99057bc7154ce1d66437ac4bc28f55bb25f7307dc194a
```

### Main baseline: three derived EF groups

VAL confusion matrix, rows recorded groups and columns predicted groups:

| Recorded / predicted | Reduced | Mildly reduced | Normal |
|---|---:|---:|---:|
| Reduced | 110 | 23 | 23 |
| Mildly reduced | 31 | 36 | 66 |
| Normal | 12 | 49 | 938 |

| VAL measure | Reduced | Mildly reduced | Normal |
|---|---:|---:|---:|
| Recall | 70.51% | 27.07% | 93.89% |
| Precision | 71.90% | 33.33% | 91.33% |

VAL accuracy is `84.16%` and macro F1 is `0.6456`. The larger Normal group lifts overall accuracy. Mildly reduced EF remains a weak group.

Two saved historical TEST group evaluations exist:

| Saved version | Correct / total | Macro F1 |
|---|---:|---:|
| Original historical group report | 1,072 / 1,277 | 0.6329 |
| Later `evaluate_all` report | 1,074 / 1,277 | 0.6367 |

Both are retained. They have different confusion counts and must not be mixed or selected according to which looks better. The exact cause of the small boundary/count changes has not been re-established.

### Supporting experiment: four-class R3D augmentation

These models learn four classes: EF below 40, 40 to below 50, 50 to below 70, and 70 or above. This is not the main app's three derived groups.

Five comparable FP32 arms used one seed, shared starting weights, a 20-epoch cap, four initially frozen-backbone epochs and patience six. VAL was divided into **643 selection / 645 reserved calibration** videos. Epochs and the augmentation winner were selected by lowest unrounded selection EF MAE, not class F1.

| Arm | Best epoch | Selection EF MAE | Four-class macro F1 |
|---|---:|---:|---:|
| MixUp | 20 | 5.6183 | 0.3003 |
| None | 18 | 5.8190 | 0.4673 |
| Temporal | 18 | 5.8491 | 0.4741 |
| CutMix | 18 | 5.9339 | 0.3742 |
| Geometric | 16 | 6.1939 | 0.4210 |

MixUp lowered selection EF MAE by `0.2007` points versus None, but its class F1 was worse. This is a modest, single-seed development result, not a new TEST result or proof that MixUp is the best classifier.

### Supporting experiment: matched Swin control versus MixUp

This pair is **regression-only EF/EDV/ESV**, with no learned four-class head. Both arms used the same 643-video selection membership, seed 42, shared initialization, FP32, batch 4, 20 completed epochs and four initially frozen-backbone epochs.

| Arm | Best epoch | Selection EF MAE |
|---|---:|---:|
| Control | 17 | 5.2355 |
| MixUp | 19 | 5.1051 |

Control minus MixUp is `0.1304` EF points. No new TEST evaluation or calibration was run for this pair. Do not compare these selection-half results directly with full-VAL or historical TEST results.

### Reliability: three different questions

| Output | Verified finding | What it does not mean |
|---|---|---|
| Main-model experimental group correctness | Logistic estimator fitted on 645 saved VAL rows and checked on the other 643: log loss 0.3145, versus 0.4400 constant and 0.3666 group-frequency baselines | Not exact EF accuracy or validated confidence on new scans |
| Original None four-class temperature calibration | Temperature 0.9017; historical TEST ECE 0.045522 to 0.035888, using 15 equal-width bins | Does not change EF or argmax decisions; cannot transfer to MixUp, Swin or the main baseline |
| Current separate residual model pair | Saved VAL calibration on 1,288 videos; nominal range level 90%; q_hat 1.7077 | Not a clinical guarantee or a range for the main baseline |

The correctness split was disjoint at the row level, but both halves were reused development data. The residual mean/head were selected using VAL, which also fitted the range. The original None four-class TEST macro F1 remains `0.4929` before and after temperature scaling; improved ECE is not improved classification.

Historical residual TEST aggregates report `90.68%` coverage and uncertainty/error Spearman `0.2299`. Deferring the highest-uncertainty 20% lowered retained-set MAE from `5.2276` to `4.7200`. Those aggregate files lack generating-checkpoint fingerprints, so they cannot be attached with certainty to the current app pair. Deferral excludes videos; it is not an improvement for all cases.

### Ordinal and explanation checks

- The frozen three-pair ordinal experiment is **Inconclusive**. Combined Mildly reduced recall rose from `25.56%` to `30.58%`, but the effect was negative for seed 303. The registered consistently-positive rule was not met.
- Separate exploratory Swin explanation checks planned 20 VAL clips. Positive Grad-CAM maps were usable on **17/20**; three empty maps remain method non-coverage.
- Masking supported sensitivity to model-used information on the usable subset, not anatomical validity. Masking can also create off-distribution inputs.
- Matched temporal alignment showed no clear advantage over matched random. The older unmatched `35.7%` versus `29.0%` comparison must not be promoted as validated improvement.

## Run history and transparency

The results above were assembled from saved metrics, experiment-completion artifacts, integrity checks and original run outputs. This README does not claim that all newer local artifacts are already committed to the public repository, or that a fresh run will reproduce an old checkpoint without its original environment.

### What was actually run (reviewed state, 8 November 2026)

| Work | Evidence location relative to the local project | Outcome |
|---|---|---|
| Main baseline VAL inference | `results/eval_split/val_r3d18_baseline_fp32_k1/metrics.json` | FP32 one-clip VAL metrics and three-group confusion |
| Historical TEST recheck | `results_recheck/baseline/recheck_results.json` | Baseline MAE 5.3793 and true R² correction |
| Original TEST group report | `results/test_eval_baseline/test_metrics.json` | Macro F1 0.6329 |
| Later TEST evaluator | `results/eval_runs/r3d18_baseline/metrics.json` | Separate MAE 5.3796 / macro F1 0.6367 version |
| Five R3D augmentation arms | `results/fourclass_fp32_v3/none/complete.json`; `results/fourclass_fp32_v4/<arm>/complete.json` | Completed selection study, MixUp lowest EF MAE |
| Matched Swin pair | `results/swin_mixup_v1/<arm>/complete.json` and `protocol.json` | Completed control and MixUp regression arms |
| Original None probability calibration | `results/fourclass_calibration_v1/report.json` and associated protocol/temperature records | Frozen checkpoint-specific calibration |
| Current residual calibration | `results/uncertainty_calibration/residual_val_calibration.json` | Separate mean/head identities and VAL calibration |
| Historical residual aggregates | `results/residual_uncertainty/residual_uncertainty_results.json` | Error-ranking/coverage evidence with missing generating identities |
| Group-correctness estimate | Saved-prediction analysis and root `group_correctness_estimate.json` v2 export | Frozen experimental estimator, identity/domain gated |
| Ordinal experiment | `results_pair_analysis/combined/combined_analysis.json` | Closed as Inconclusive |
| Local app | `app/app.py` and accepted laptop runs | TEST-sample interface; separate reference and influence views |

These paths identify evidence and local runtime dependencies. They are not download links, a promise of public availability or instructions to commit their whole folders.

### Deviations from the original plan (honest notes)

1. **The main output is regression plus three derived groups.** It is not the originally proposed learned four-class main interface. Learned four-class experiments exist separately.
2. **The input recipe changed.** The implemented path uses 32 stride-2 frames rather than 16 uniformly sampled frames. Swin uses 112 resolution rather than 224. Separate head/backbone optimizer groups do not establish layerwise learning-rate decay.
3. **Original baseline provenance is incomplete.** The checkpoint alone does not establish the exact original training code/data version. Normalization is supplied by the evaluator constants, not stored in that checkpoint.
4. **Old metric labels and duplicate results needed correction.** Squared Pearson correlation was previously labelled R². A historical balanced result duplicated the baseline and was removed from the reported comparison. The corrected evidence is retained rather than rewriting what happened.
5. **There are two TEST group-report versions.** Both remain labelled, with their own counts. The larger F1 is not silently substituted.
6. **Some experiments failed or were negative.** Earlier AMP failures are not comparable completed FP32 augmentation arms. MC Dropout did not provide useful error ranking in the historical check; heteroscedastic sigma was constant. Ordinal results were inconclusive and matched explanation alignment did not establish an advantage.
7. **Calibration is checkpoint-specific, not universal.** The original None four-class calibration cannot be assigned to an augmentation winner or a regression-only Swin model. Optional winner calibration remains paused.
8. **Deployment scope is local.** The accepted app is not a public service, external API, ONNX export or Docker release. Report and presentation drafts exist, but final repository packaging remains pending. A poster is not part of the current deliverables.
9. **EfficientNet-B3 + LSTM is deferred.** A frame CNN with mean pooling is not that architecture. No B3+LSTM result is claimed.

### Raw data and figures

Per-video records, prediction arrays, dataset IDs, cached clips and dataset-derived example images remain local and must not be bundled with a new code release. Aggregate tables and safe figures can be prepared from the verified evidence after review. This file does not link to private archives or claim a public figures package exists.

## Data and models

### Dataset access and local configuration

Get EchoNet-Dynamic through its official site: <https://echonet.github.io/dynamic/>. Follow its access process and current terms. Having a local copy does not authorize redistributing the videos or dataset-derived records.

Set these entries in `configs/base.yaml`. The following are project-relative example locations, not someone's private machine paths:

```yaml
data:
  raw_video_dir: "data/EchoNet-Dynamic/Videos"
  file_list_csv: "data/EchoNet-Dynamic/FileList.csv"
  volume_tracings_csv: "data/EchoNet-Dynamic/VolumeTracings.csv"
```

The TEST selector needs original videos and a `FileList.csv` containing at least `FileName`, `Split` and `EF`. `VolumeTracings.csv` supports annotation-based research; it is not needed just to select a TEST video. Cached training clips are not necessary for the app's raw-video path.

No patient IDs or validated age/sex fields are available in the working evaluation metadata. Video-level split checks are not patient-independence or demographic-fairness checks.

### Local runtime artifacts

| Purpose | Expected project-relative location | Required for |
|---|---|---|
| Main baseline weights | `checkpoints/r3d18_best.pt` | Main prediction |
| Saved baseline VAL metrics | `results/eval_split/val_r3d18_baseline_fp32_k1/metrics.json` | Baseline reference and estimator identity checks |
| Experimental estimator v2 | `group_correctness_estimate.json` | Displaying an eligible group-correctness estimate |
| Separate residual mean model | `checkpoints/r3d18_mcdropout_best.pt` | Residual research experiment |
| Separate residual error head | `checkpoints/r3d18_residual_uncertainty_best.pt` | Residual research experiment |
| Residual calibration | `results/uncertainty_calibration/residual_val_calibration.json` | Residual research experiment |

Keep the exact matching sets. A missing optional artifact must not be replaced with guessed numbers or a different model. Only load trusted original checkpoints. No checkpoint binaries or prediction archives are supplied with this README.

### Repository history and licensing

The existing public README records that older commits, before `40c957c`, contain per-video prediction files and three Grad-CAM example images derived from the dataset. **Do not treat the Git history as free of dataset-derived material.** Adding ignore rules or deleting files in a later commit does not remove earlier history. Release cleanup must address this before calling the repository distribution-ready.

No project license has been chosen in the reviewed public repository. This README does not grant a new code, model or dataset license. Dataset access and use remain subject to their own terms.

### Citations

These are method and dataset references, not claims that the project achieved the papers' published results.

- Ouyang et al. *Video-based AI for beat-to-beat assessment of cardiac function*. Nature, 2020. <https://www.nature.com/articles/s41586-020-2145-8>
- EchoNet-Dynamic dataset and access information. <https://echonet.github.io/dynamic/>
- Hara, Kataoka and Satoh. *Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?* CVPR, 2018. <https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html>
- Liu et al. *Video Swin Transformer*. CVPR, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Liu%5FVideo%5FSwin%5FTransformer%5FCVPR%5F2022%5Fpaper.html>
- Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>
- Guo et al. *On Calibration of Modern Neural Networks*. ICML, 2017. <https://proceedings.mlr.press/v70/guo17a.html>
- Angelopoulos and Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv, 2021. <https://arxiv.org/abs/2107.07511>
- Zhang et al. *mixup: Beyond Empirical Risk Minimization*. <https://arxiv.org/abs/1710.09412>
- Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV, 2019. <https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html>

## Known limitations

- Single dataset, no external validation and no validated new-scan performance.
- VAL and TEST were used during development. There is no untouched final test result here.
- Patient independence cannot be checked from the available metadata; bootstrap intervals are not patient-clustered guarantees.
- Mildly reduced EF has weak precision and recall. Normal-dominated accuracy must not hide that failure.
- No demographic-fairness assessment, validated automatic view/quality check or whole-video coverage guarantee.
- A single centered clip can miss relevant parts of a video. Bounds checks are not clinical safety validation.
- Calibration is not completed for every model. Regression does not provide class probabilities to temperature-scale directly.
- Uncertainty and explanation results have method-specific limits. Historical residual aggregates have incomplete model identity; attribution maps are not anatomy labels.
- Newer augmentation and matched Swin results use one seed and reused selection data. They do not establish general superiority.
- Mac end-to-end support and clean-machine Windows setup are not verified. The accepted Windows environment is not yet a reproducible public installer.
- No public checkpoint download, automated data fetch, public deployment, finished API/ONNX/Docker package or B3+LSTM result is claimed.

## Project rules

- Label every number by model, split, evaluation version and output type. Do not mix regression, derived groups and learned four-class scores.
- Keep the original R3D-18 baseline as the main app model unless a separately reviewed change explicitly replaces it.
- Use recorded EF only as a reference after prediction, never as predictor input.
- Keep calibration tied to the checkpoint, preprocessing and fit data that produced it. Do not claim clean separation retrospectively for reused development splits.
- Withhold unsupported outputs. Do not bypass missing-artifact, identity or fitted-domain checks to display a confidence percentage.
- Retain weak, failed and inconclusive results. Deferred work is not completed work.
- Keep data, checkpoints, prediction archives, video IDs, private paths and credentials out of new source publications. Review Git history as well as the current tree.
- Run new training, evaluation or calibration only as a separate, explicit task. Launching the demo does not require restarting experiments.
- Keep every claim within academic research scope. No medical advice, diagnostic claim or safety guarantee.
- Disclose assistance honestly. AI tools assisted coding, debugging, evidence review and document preparation; the author remains responsible for checking the code, understanding the methods and verifying the results. Follow the institution's actual disclosure requirements.
