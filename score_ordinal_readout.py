"""Secondary readout scorer for the ordinal arm (pre-registered experiment).

NEW FILE. Scores ONLY the ordinal head's four-class readout on VAL with the
same centred-clip sampling and the same preprocessing the unchanged evaluator
uses: clips arrive from the dataset in [0, 1] and are normalized per channel
with the mean/std resolved from the checkpoint by the evaluator's own
resolve_normalisation (R3D Kinetics constants unless the checkpoint saves
norm_mean/norm_std). This is the pre-registered SECONDARY result; the primary
comparison stays the EF regression output thresholded at 40/50, scored
through the unchanged evaluator on stripped checkpoints. This scorer is never
promoted to primary.

Scope: VAL ONLY under the current authority. The single frozen, disclosed
TEST scoring at the end of the experiment is a separate, explicitly approved
step and is not enabled here.

Fixed four groups (from the ordinal head, >= at each cut-off, independent of
the config's two thresholds): <40, 40-50, 50-70, >=70. Metrics are fixed by
the pre-registration: per-group recall, overall accuracy, and the full
confusion matrix. Any VAL group with fewer than 30 videos is flagged as
exploratory (same rule as the group-count script).

Safety: checkpoint loads with weights_only=True only (no fallback); the
checkpoint's saved settings (num_frames, frame_stride, ef_thresholds) are
compared with the config by the evaluator's own helpers and any mismatch
stops the run; the model runs in eval() mode in FP32 (TF32 disabled, as in
the evaluator); results go to a NEW folder results_ordinal_readout/<tag>/
which must not already exist. No training, no checkpoint changes.

Run from the project root (only after an approved run produces a checkpoint):
    python score_ordinal_readout.py --config configs/base.yaml --checkpoint checkpoints\r3d18_ordinal_seed101\best.pt --tag ordinal_seed101_val
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.eval.evaluate_split import (
    compare_checkpoint_settings,
    find_checkpoint_settings,
    resolve_normalisation,
)
from src.models.ordinal_head import ef_to_ordinal_class, predict_class
from src.models.r3d_multitask_ordinal import R3DMultiTaskOrdinal

GROUP_NAMES = ["<40", "40-50", "50-70", ">=70"]
EXPLORATORY_MIN = 30  # same rule as count_ef_groups.py


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score(config_path: str, checkpoint: str, tag: str,
          num_workers: int = 0, batch_size: int = None) -> dict:
    """VAL-only scoring of the ordinal four-class readout."""
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = Path("results_ordinal_readout") / tag
    if out_dir.exists():
        raise SystemExit(f"STOPPED: {out_dir} already exists. Choose a new --tag (results are never overwritten).")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"STOPPED: checkpoint not found: {ckpt_path}")
    ckpt_hash = _sha256_of(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise SystemExit(f"STOPPED: torch.load(weights_only=True) failed for {ckpt_path}:\n"
                         f"{type(e).__name__}: {e}\nNo unsafe fallback is provided.")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit("STOPPED: checkpoint is not a dict with 'model_state_dict'")

    # same compatibility checks the unchanged evaluator applies
    found = find_checkpoint_settings(ckpt)
    setting_rows, mismatches = compare_checkpoint_settings(found, cfg["data"])
    if mismatches:
        raise SystemExit("STOPPED: checkpoint settings disagree with the evaluation config:\n  "
                         + "\n  ".join(mismatches))
    try:
        mean, std, norm_source, norm_gaps = resolve_normalisation("r3d18", ckpt)
    except ValueError as e:
        raise SystemExit(f"STOPPED: {e}")
    for r in setting_rows:
        if r["status"] == "not saved in checkpoint":
            print(f"  provenance gap: {r['setting']}: not saved in checkpoint, so it could not "
                  f"be checked against the evaluation config")
    for g in norm_gaps:
        print(f"  provenance gap: {g}")

    model = R3DMultiTaskOrdinal(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    try:
        model.load_state_dict(ckpt["model_state_dict"])  # strict: ordinal keys must be present
    except RuntimeError as e:
        raise SystemExit(f"STOPPED: checkpoint does not match the ordinal architecture "
                         f"(missing or unexpected keys). Use the full ordinal checkpoint, "
                         f"not a stripped one.\n{e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()  # eval mode: dropout off, deterministic given the weights
    torch.backends.cudnn.allow_tf32 = False      # FP32, as in the evaluator's fp32 mode
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
    loader = DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"],
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    # preprocessing identical to the evaluator: [0,1] clip -> per-channel normalization
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)

    filenames, true_efs, true_cls, pred_cls = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            clip = (batch["clip"].to(device) - mean_t) / std_t
            out = model(clip)
            filenames.extend(batch["filename"])
            true_efs.extend(batch["ef"].tolist())
            true_cls.extend(ef_to_ordinal_class(batch["ef"]).tolist())
            pred_cls.extend(predict_class(out["ordinal_logits"].cpu()).tolist())

    n = len(filenames)
    confusion = [[0] * 4 for _ in range(4)]
    for t, p in zip(true_cls, pred_cls):
        confusion[t][p] += 1
    per_group = {}
    for i, name in enumerate(GROUP_NAMES):
        n_i = sum(confusion[i])
        recall_i = (confusion[i][i] / n_i) if n_i else None
        per_group[name] = {"n": n_i, "recall": recall_i,
                           "exploratory": bool(n_i and n_i < EXPLORATORY_MIN)}
    correct = sum(confusion[i][i] for i in range(4))
    metrics = {
        "split": "VAL", "n_videos": n,
        "checkpoint": str(ckpt_path), "checkpoint_sha256": ckpt_hash,
        "readout": "ordinal head four-class (secondary); thresholds 40/50/70, >= at each",
        "preprocessing": {
            "normalization_mean": mean, "normalization_std": std,
            "normalization_source": norm_source,
            "sampling": "single centred clip per video, no temporal augmentation "
                        "(same as the unchanged evaluator)",
            "precision": "fp32", "device": str(device),
        },
        "overall_accuracy": correct / n if n else None,
        "per_group": per_group,
        "confusion_true_rows_pred_cols": confusion,
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ef", "true_class", "pred_class"])
        w.writerows(zip(filenames, true_efs, true_cls, pred_cls))

    print(f"VAL: n={n} | overall accuracy (4-class) = {metrics['overall_accuracy']:.3f}")
    print(f"preprocessing: normalisation [{norm_source}], fp32, {device}")
    for name in GROUP_NAMES:
        g = per_group[name]
        rec = "n/a" if g["recall"] is None else f"{g['recall']:.3f}"
        flag = " [exploratory: < 30 videos]" if g["exploratory"] else ""
        print(f"  group {name}: n={g['n']} recall={rec}{flag}")
    print(f"Saved {out_dir / 'metrics.json'} and {out_dir / 'predictions.csv'}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    score(args.config, args.checkpoint, args.tag,
          num_workers=args.num_workers, batch_size=args.batch_size)

