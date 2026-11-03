"""Pre-registered equivalence check: does stripping ordinal_head.* keys change
the EF regression output? NEW FILE. Runs ONCE after the first real ordinal
run finishes, before any seed is scored.

Two levels, fixed in advance:
  (a) WEIGHT-LEVEL (must pass EXACTLY): every baseline-key tensor in the
      stripped checkpoint equals the corresponding tensor in the unstripped
      ordinal checkpoint (torch.equal). Any failure means the converter
      misbehaved - stop, do not score.
  (b) PREDICTION-LEVEL: both checkpoints are loaded (unstripped into
      R3DMultiTaskOrdinal, stripped into R3DMultiTask), BOTH in eval() mode,
      same device, FP32 enforced (TF32 disabled), and run on the FIRST 20
      VAL videos in FileList.csv order (fixed subset, no random selection).
      Pass criteria: max absolute EF difference <= 1e-4 AND zero flips of
      the 40/50 classifications. Exact identity is NOT demanded: GPU/cuDNN
      nondeterminism can move predictions slightly even for identical
      weights, so a small fixed tolerance is pre-registered. The
      pre-registered route is CPU.

The stripped file's metadata (config, epoch, val_mae) must equal the ordinal
checkpoint's, and normalisation is resolved on BOTH files independently (the
results must agree) - the stripped file is not trusted to be a converter
product without these checks.

Both checkpoint hashes, the video list and the observed max difference are
recorded to a NEW JSON file (--out, must not already exist). No training, no
split evaluation beyond the fixed 20-video check. The pre-registered route is
CPU (the default); FP32 is enforced on any device by disabling TF32, and both
models always run on the SAME device.

Run from the project root:
    python check_stripped_equivalence.py checkpoints\r3d18_ordinal_seed101\best.pt checkpoints\r3d18_ordinal_seed101\best_for_eval.pt --config configs/base.yaml --out checkpoints\r3d18_ordinal_seed101\equivalence_check.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import math

import torch
import yaml

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.eval.evaluate_split import (
    compare_checkpoint_settings,
    find_checkpoint_settings,
    resolve_normalisation,
)
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_multitask_ordinal import R3DMultiTaskOrdinal
from strip_ordinal_keys import ORDINAL_KEY_ALLOWLIST

N_VIDEOS = 20
MAX_ABS_EF_DIFF = 1e-4
MAX_CLASS_FLIPS = 0
EF_THRESHOLDS = (40.0, 50.0)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise SystemExit(f"STOPPED: torch.load(weights_only=True) failed for {path}:\n"
                         f"{type(e).__name__}: {e}\nNo unsafe fallback is provided.")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit(f"STOPPED: {path} is not a dict with 'model_state_dict'")
    return ckpt


def check(ordinal_ckpt_path: str, stripped_ckpt_path: str, config_path: str,
          out_json: str, n_videos: int = N_VIDEOS, device: str = "cpu") -> dict:
    ord_p, str_p, out_p = Path(ordinal_ckpt_path), Path(stripped_ckpt_path), Path(out_json)
    if out_p.exists():
        raise SystemExit(f"STOPPED: {out_p} already exists; refusing to overwrite a recorded check")
    ord_ckpt, str_ckpt = _load(ord_p), _load(str_p)

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    # same compatibility checks the unchanged evaluator applies (ordinal ckpt is the source)
    found = find_checkpoint_settings(ord_ckpt)
    _, mismatches = compare_checkpoint_settings(found, cfg["data"])
    if mismatches:
        raise SystemExit("STOPPED: checkpoint settings disagree with the config:\n  "
                         + "\n  ".join(mismatches))
    try:
        mean, std, norm_source, _ = resolve_normalisation("r3d18", ord_ckpt)
    except ValueError as e:
        raise SystemExit(f"STOPPED: {e}")

    # the stripped file must carry identical metadata and resolve the same norm
    for field in ("epoch", "val_mae", "config"):
        if ord_ckpt.get(field) != str_ckpt.get(field):
            raise SystemExit(f"STOPPED: metadata mismatch on '{field}' between the ordinal and "
                             f"stripped checkpoints; the stripped file is not a faithful copy")
    try:
        s_mean, s_std, s_source, _ = resolve_normalisation("r3d18", str_ckpt)
    except ValueError as e:
        raise SystemExit(f"STOPPED: {e}")
    if s_mean != mean or s_std != std or s_source != norm_source:
        raise SystemExit("STOPPED: normalisation resolution differs between the ordinal and "
                         "stripped checkpoints; do not score")

    # (a) weight-level: exact equality on every baseline key
    ord_sd, str_sd = ord_ckpt["model_state_dict"], str_ckpt["model_state_dict"]
    if set(str_sd) != set(ord_sd) - set(ORDINAL_KEY_ALLOWLIST):
        raise SystemExit("STOPPED: stripped checkpoint's keys are not exactly the ordinal "
                         "checkpoint's keys minus the allowlist; do not score")
    for k in str_sd:
        if not torch.equal(ord_sd[k], str_sd[k]):
            raise SystemExit(f"STOPPED: weight mismatch on '{k}' between the ordinal and "
                             f"stripped checkpoints; the converter misbehaved - do not score")
    print(f"PASS (weights): all {len(str_sd)} baseline-key tensors are exactly equal")

    # (b) prediction-level on the fixed VAL subset
    m_ord = R3DMultiTaskOrdinal(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    m_ord.load_state_dict(ord_ckpt["model_state_dict"])
    m_base = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    m_base.load_state_dict(str_ckpt["model_state_dict"])
    dev = torch.device(device)
    m_ord.to(dev).eval()   # eval mode in BOTH: dropout off
    m_base.to(dev).eval()
    torch.backends.cudnn.allow_tf32 = False      # FP32 enforced on any device
    torch.backends.cuda.matmul.allow_tf32 = False

    ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"],
        file_list_csv=cfg["data"]["file_list_csv"],
        split="VAL",
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"],
        augment=AugmentConfig(temporal=False),
        seed=cfg["seed"],
    )
    n_take = min(n_videos, len(ds))
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1, 1)
    max_diff, flips, per_video = 0.0, 0, []
    with torch.no_grad():
        for i in range(n_take):
            item = ds[i]
            # preprocessing identical to the evaluator: [0,1] clip -> per-channel norm
            clip = (item["clip"].unsqueeze(0).to(dev) - mean_t) / std_t
            ef_ord = m_ord(clip)["ef"].item()
            ef_base = m_base(clip)["ef"].item()
            if not (math.isfinite(ef_ord) and math.isfinite(ef_base)):
                raise SystemExit(f"STOPPED: non-finite EF prediction on {item['filename']} "
                                 f"(ordinal={ef_ord}, stripped={ef_base}); do not score")
            d = abs(ef_ord - ef_base)
            max_diff = max(max_diff, d)
            c_ord = sum(ef_ord >= t for t in EF_THRESHOLDS)
            c_base = sum(ef_base >= t for t in EF_THRESHOLDS)
            flips += int(c_ord != c_base)
            per_video.append({"filename": item["filename"], "ef_ordinal": ef_ord,
                              "ef_stripped": ef_base, "abs_diff": d})

    ok = max_diff <= MAX_ABS_EF_DIFF and flips <= MAX_CLASS_FLIPS
    record = {
        "ordinal_checkpoint": str(ord_p), "ordinal_sha256": _sha256_of(ord_p),
        "stripped_checkpoint": str(str_p), "stripped_sha256": _sha256_of(str_p),
        "device": device, "precision": "fp32",
        "preprocessing": {"normalization_mean": mean, "normalization_std": std,
                          "normalization_source": norm_source,
                          "sampling": "single centred clip per video, no temporal augmentation"},
        "n_videos": n_take,
        "videos": [v["filename"] for v in per_video],
        "max_abs_ef_diff": max_diff, "tolerance": MAX_ABS_EF_DIFF,
        "threshold_flips_40_50": flips, "allowed_flips": MAX_CLASS_FLIPS,
        "weight_level": "exact equality on all baseline keys: PASS",
        "metadata_check": "config/epoch/val_mae identical and normalisation resolution "
                          "agrees on both checkpoints: PASS",
        "result": "PASS" if ok else "FAIL",
        "per_video": per_video,
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    status = "PASS" if ok else "FAIL - investigate the forward path before any scoring"
    print(f"{'PASS' if ok else 'FAIL'} (predictions): max |EF difference| = {max_diff:.3e} "
          f"(tolerance {MAX_ABS_EF_DIFF}), 40/50 flips = {flips} (allowed {MAX_CLASS_FLIPS})")
    print(f"{status}. Record written to {out_p}")
    if not ok:
        raise SystemExit("STOPPED: prediction-level equivalence failed; do not score until resolved")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ordinal_checkpoint")
    parser.add_argument("stripped_checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n-videos", type=int, default=N_VIDEOS)
    parser.add_argument("--device", type=str, default="cpu",
                        help="both models always run on the SAME device; cpu is the deterministic default")
    args = parser.parse_args()
    check(args.ordinal_checkpoint, args.stripped_checkpoint, args.config, args.out,
          n_videos=args.n_videos, device=args.device)
