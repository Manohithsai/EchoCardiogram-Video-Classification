# Echo-EF: Ejection Fraction Estimation from Echocardiogram Video

Research code for estimating left ventricular ejection fraction (LVEF) from
apical four-chamber echocardiogram videos on the EchoNet-Dynamic dataset. The
project compares video and frame models, tests uncertainty estimates, and runs
one pre-registered experiment on an ordinal head. It is a student research
project, not a clinical tool.

## Current status

- Trained and evaluated: R3D-18 (several loss variants), a frame-level CNN, and Video Swin-T, all with EF/EDV/ESV multi-task outputs.
- Uncertainty: MC Dropout, heteroscedastic output, and a residual-uncertainty head, each with conformal intervals.
- Explainability: 3D Grad-CAM with a check against annotated key frames.
- Pre-registered ordinal-head experiment: finished, result **Inconclusive** (see below).
- Not built yet: ONNX export, serving API, Docker image, demo app, paper.
- Planned, not started: an EfficientNet-B3 + LSTM comparison. Its pre-registration is still in review and nothing has been trained.

## Data

Uses [EchoNet-Dynamic](https://echonet.github.io/dynamic/) (Stanford AIMI). You
must register and accept the Research Use Agreement yourself. **The current version
of this repository contains no dataset files.** It leaves out per-video prediction
files, video-ID lists and example frames, and keeps only code, aggregate metrics and
summary plots. **Older commits in the Git history (before 40c957c) still contain
per-video prediction files and three Grad-CAM example images derived from the dataset.**
Do not treat the history as free of dataset-derived files. Splits: TRAIN 7,465 / VAL 1,288 / TEST 1,277 videos.

## Setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate    Windows: .venv\Scripts\activate
# First install torch and torchvision for your CUDA version (pytorch.org)
pip install -r requirements.txt
```

Point `configs/base.yaml` (`data.raw_video_dir`, `data.file_list_csv`,
`data.volume_tracings_csv`) at your download, then decode the videos once:

```bash
python scripts/build_cache.py --config configs/base.yaml
python -m pytest tests/ -v
```

Developed on an RTX 3060 Laptop GPU (6 GB), so defaults are small (batch 4, 112x112 frames, 32 frames per clip, mixed precision).

## Training and evaluation

```bash
python -m src.train.train --config configs/base.yaml            # R3D-18 baseline
python -m src.train.train_swin3d --config configs/base.yaml     # Video Swin-T
python -m src.eval.evaluate_split --config configs/base.yaml --checkpoint checkpoints/r3d18_best.pt --tag my_val_run --arch r3d18 --split VAL
```

`evaluate_split` writes to a new folder `results/eval_split/<tag>/` (it refuses to overwrite) and also saves per-video
files there; keep those local and do not commit them. Use `--arch swin3d_t` for Swin.

Other training scripts (`train_r3d_lds`, `train_r3d_balanced`, `train_frame_cnn`,
`train_heteroscedastic`, `train_residual_uncertainty`, `train_matched_baseline`,
`train_ordinal`) live in `src/train/`; check each script's `--help` for its arguments.

## Results

TEST was used during development, so the TEST numbers below are verification numbers, not an
untouched final test. Ordinal-experiment numbers use the VAL split. **No trained checkpoints are included and there is no download link.** To reproduce the numbers, train the models first with the scripts below, or point `--checkpoint` at your own compatible checkpoint.

| Model (TEST, 1,277 videos) | MAE | RMSE | R2 |
|---|---|---|---|
| R3D-18 baseline | 5.379 | 7.345 | 0.639 |
| R3D-18 + LDS | 5.962 | 7.690 | 0.604 |
| R3D-18 + class-balanced DRW | 5.662 | 7.742 | 0.599 |
| Frame-CNN | 6.222 | 8.652 | 0.499 |
| Video Swin-T | 5.385 | 7.224 | 0.651 |

MAE and RMSE are in EF percentage points (lower is better). R2 is the coefficient of determination (higher is better, 1 is perfect). Observed MAE is similar for Swin-T and R3D-18. In an exploratory paired bootstrap the MAE difference
(Swin minus R3D) is +0.005 with a 95% interval of -0.228 to +0.229, which crosses zero, so this
comparison shows no clear winner. It does not show the two are equivalent.
An earlier results file reported R3D-18 R2 as 0.647; that was the squared Pearson correlation.
`results_recheck/` holds the inference-only recheck and `AUDIT_SUMMARY.txt`.
It also records that an earlier "balanced" result file was a duplicate of the baseline,
so that file was removed from the repo.

Uncertainty (saved result files, TEST split; conformal intervals calibrated on VAL, 90% target):
coverage is 90.8% (MC Dropout), 91.5% (heteroscedastic) and 90.7% (residual head). MC Dropout
does not rank errors (Spearman -0.04, p=0.20). The heteroscedastic sigma is constant, so it
cannot rank errors. The residual head does rank errors (Spearman 0.23);
deferring the 20% most uncertain cases lowers MAE from 5.23 to 4.72. The saved
uncertainty files do not record which checkpoint produced them.

Grad-CAM check (300 samples, 15-frame tolerance against annotated key frames):
hit rate 35.7% versus 29.0% for a random baseline. That is a weak signal.

### Pre-registered ordinal experiment (closed)

Question: does an ordinal head raise recall of mildly reduced EF (40-49) without
hurting normal-EF recall or MAE? Three seeds, VAL split, one centred clip,
analysis frozen before seed 303 ran (`docs/ordinal_experiment_prereg.md`,
`docs/prereg_addendum_combined.md`, `compare_combined.py`).

| Combined, 3 seeds | Baseline | Ordinal | Difference (95% interval) |
|---|---|---|---|
| Mild recall (133 videos) | 25.6% | 30.6% | +5.0 pp (+0.8 to +9.8) |
| Normal recall (999 videos) | 95.2% | 94.2% | -1.0 pp (-1.7 to -0.3) |
| MAE (1,288 videos) | 5.534 | 5.416 | -0.118 (-0.218 to -0.017) |

Normal recall stayed above the registered 92% guardrail, and the MAE guardrail also passed. Label: **Inconclusive.** Mild recall rose in seeds 101 and 202 but fell in 303, so
the frozen "consistently positive" rule was not met, and the MAE gain comes mostly
from seed 202. Intervals reflect video sampling, not training-run variation.

## Limitations

- Single dataset (EchoNet-Dynamic). No external validation.
- TEST was used during development. No untouched final test.
- Patient independence of splits is not checked (the dataset has no patient IDs).
- Ensembles and multi-clip evaluation were picked on VAL, so they are exploratory.
- The original R3D-18 training code version is not fully recorded; a fresh matched baseline was trained for the ordinal experiment.
- Not validated for clinical use.

## Repository layout

```
src/            data, models, training, evaluation, explainability
scripts/        cache builder
configs/        base.yaml
tests/ + test_*.py   unit and integration tests
docs/           pre-registration, addendum, launch log
analysis_archive_2026-10-29/   exploratory VAL analyses (kept for provenance)
results*/       aggregate metrics and plots; no per-video files
```

## License

No license has been chosen yet. Until one is added, default copyright applies.
The dataset has its own terms (non-commercial research use, no redistribution).
