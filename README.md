# Echo-EF: Uncertainty-Aware Ejection Fraction Estimation

Video deep learning for LVEF estimation from echocardiograms, with
multi-task EDV/ESV prediction, calibrated uncertainty (MC Dropout +
conformal prediction), and validated Grad-CAM explanations.

## Status
Repo scaffold + R3D-18 multi-task model + dataset pipeline. Training,
uncertainty, XAI, and serving code are the next steps (see roadmap below).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` intentionally omits pinned CUDA builds of torch —
install the torch/torchvision build matching your CUDA version separately
(see pytorch.org), then `pip install -r requirements.txt` for the rest.

## Data access
This project uses **EchoNet-Dynamic** (Stanford). It requires registering
with Stanford AIMI and accepting a non-commercial Research Use Agreement.
1. Register: https://echonet.github.io/dynamic/
2. Download `Videos/`, `FileList.csv`, `VolumeTracings.csv`
3. **Do not commit the data or push it to this repo** — only code and
   results go in version control; the RUA is non-commercial research only.

Point `configs/base.yaml`'s `data.raw_video_dir` and `data.file_list_csv`
at wherever you put the download.

## Pipeline
```bash
# 1. One-time: decode AVIs into a fast-loading uint8 cache (resumable)
python scripts/build_cache.py --config configs/base.yaml

# 2. Sanity-check the cache/labels logic without touching real data
python -m pytest tests/ -v

# 3. Train (coming next)
python -m src.train.train --config configs/base.yaml
```

## Why a cache step
Decoding AVI on the fly is the actual bottleneck on a laptop GPU, not the
model forward pass. Caching once to `.npy` makes every epoch after the
first run fast, and the step is resumable (re-running skips files that are
already cached).

## Hardware notes
Developed against an RTX 3060 Laptop GPU (6GB VRAM). Defaults in
`configs/base.yaml` (batch size 4, 112×112, mixed precision) are set
conservatively for that card. Video Swin fine-tuning is intended to run on
Kaggle's free GPU tier, not locally — see `docs/` (planned) for the Kaggle
notebook.

## Roadmap
- [x] Repo scaffold, config, cache script, dataset class, R3D-18 multi-task model
- [ ] Training loop (frame-CNN baseline, R3D-18, EF-consistency loss ablation)
- [ ] MC Dropout + conformal prediction, selective-prediction evaluation
- [ ] Grad-CAM 3D + temporal occlusion, validated against ED/ES annotations
- [ ] Video Swin-T fine-tuning (Kaggle)
- [ ] ONNX export, FastAPI service, Docker, latency benchmark, Gradio demo
- [ ] Extensions (time permitting): deep ensemble, robustness/corruption testing, CAMUS external validation

## License / data use
Code: MIT (adjust as needed). Data: EchoNet-Dynamic is not redistributed
here and is subject to Stanford AIMI's non-commercial Research Use
Agreement — see their site for terms.
