"""
MATCHED-BASELINE trainer for the pre-registered experiment. NEW FILE copied
from src/train/train.py, which is NOT edited. It contains NO ordinal
machinery. Differences, deliberately minimal:

1. Optional --seed override so each pre-registered seed is set per run and
   recorded; the ordinal arm (src/train/train_ordinal.py) sets the SAME
   seeds through the identical flag, so the two arms differ ONLY by the
   ordinal head. The ordinal arm preserves the CPU RNG state around its
   ordinal head's initialization (torch.random.fork_rng in the model file),
   so at the same seed both arms leave model construction with the same
   RNG stream position.
2. Checkpoints save under a per-seed run directory,
   checkpoints/r3d18_matched_baseline_seed<seed>/best.pt, created with a
   fail-if-exists guard so a recorded run is never overwritten;
   r3d18_best.pt is never touched. A provenance.json (file hashes, config,
   normalization constants, library versions, seed, wandb mode, timestamp)
   is written into the run directory at start.
3. The wandb run name is f"r3d18-matched-baseline-seed<seed>" and logging
   is forced offline (mode="offline" at wandb.init), so nothing is uploaded
   regardless of environment settings; syncing logs to the W&B service is a
   separate explicit owner decision.

Everything else is byte-identical to the baseline trainer.

This file is preparation only; running any training is a separate,
explicitly approved step.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Logging is forced offline at wandb.init (mode="offline"): nothing is
# uploaded to the W&B service regardless of environment settings. Syncing
# logs later is a separate explicit owner decision.
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask, ef_consistency_loss

# R3D-18 was pretrained on Kinetics-400 with these normalization stats.
# Using ImageNet stats here would silently hurt transfer learning.
R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_provenance(run_dir: Path, cfg: dict, arm_files: list) -> None:
    """Record what this run executed: file hashes, config, normalization
    constants, library versions, seed, wandb mode and timestamp."""
    root = Path(__file__).resolve().parents[2]
    file_hashes = {}
    for rel in arm_files:
        p = root / rel
        file_hashes[rel] = _sha256_of(p) if p.is_file() else "not found"
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for mod_name in ("torchvision", "numpy", "pandas", "wandb", "yaml"):
        try:
            versions[mod_name] = getattr(__import__(mod_name), "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    provenance = {
        "seed": cfg["seed"],
        "config": cfg,
        "file_sha256": file_hashes,
        "normalization": {"mean": R3D_MEAN, "std": R3D_STD},
        "library_versions": versions,
        "wandb_mode": "offline (forced at init)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = run_dir / "provenance.json"
    out.write_text(json.dumps(provenance, indent=2, default=str))
    print(f"Provenance written to {out}")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip: torch.Tensor, mean: list, std: list, device: torch.device) -> torch.Tensor:
    """clip: (B, 3, T, H, W) float32 in [0, 1] -> normalized."""
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def compute_loss(out: dict, ef: torch.Tensor, edv: torch.Tensor, esv: torch.Tensor,
                  consistency_weight: float) -> torch.Tensor:
    loss_ef = nn.functional.smooth_l1_loss(out["ef"], ef)
    loss_edv = nn.functional.smooth_l1_loss(out["edv"], edv)
    loss_esv = nn.functional.smooth_l1_loss(out["esv"], esv)
    loss = loss_ef + loss_edv + loss_esv
    if consistency_weight > 0:
        loss = loss + consistency_weight * ef_consistency_loss(out["ef"], out["edv"], out["esv"])
    return loss


def run_epoch(model, loader, optimizer, scaler, cfg, device, train: bool) -> tuple:
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    use_amp = cfg["train"]["amp"]

    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = batch["clip"].to(device, non_blocking=True)
            clip = normalize(clip, R3D_MEAN, R3D_STD, device)
            ef = batch["ef"].to(device)
            edv = batch["edv"].to(device)
            esv = batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(clip)
                loss = compute_loss(out, ef, edv, esv, consistency_weight)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item()
            if np.isfinite(loss_val):
                total_loss += loss_val
                total_mae += (out["ef"].detach() - ef).abs().mean().item()
                n_batches += 1
            else:
                print(f"  [skip] non-finite loss encountered (AMP scaler still warming up)")

        return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)


def build_optimizer(model: R3DMultiTask, cfg: dict) -> torch.optim.Optimizer:
    """Separate, lower LR for the pretrained backbone than for the new head --
    protects the pretrained features from being wrecked by early large updates."""
    return torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": cfg["train"]["lr_head"]},
            {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override the config seed; record the value used (pre-registered seeds)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        print(f"Seed overridden by --seed: {cfg['seed']}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available -- training will be very slow on CPU.")

    wandb.init(project="echo-ef", config=cfg, name=f"r3d18-matched-baseline-seed{cfg['seed']}",
               mode="offline")  # forced: no uploads without a separate explicit owner decision

    # --- Data ---
    train_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"],
        file_list_csv=cfg["data"]["file_list_csv"],
        split="TRAIN",
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"],
        augment=AugmentConfig(temporal=True),
        seed=cfg["seed"],
    )
    val_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"],
        file_list_csv=cfg["data"]["file_list_csv"],
        split="VAL",
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"],
        augment=AugmentConfig(temporal=False),
        seed=cfg["seed"],
    )
    print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    # --- Model ---
    model = R3DMultiTask(
        pretrained=cfg["model"]["pretrained"],
        dropout_p=cfg["model"]["dropout_p"],
    ).to(device)

    # Warm-up: freeze backbone for the first N epochs so the randomly
    # initialized head doesn't wreck the pretrained features.
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"], init_scale=2.0**12)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    run_dir = ckpt_dir / f"r3d18_matched_baseline_seed{cfg['seed']}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(
            f"Run directory {run_dir} already exists. Refusing to overwrite a "
            f"recorded run: use the next pre-registered seed, or delete the "
            f"directory deliberately if this exact run is being repeated."
        )
    _write_provenance(run_dir, cfg, ["src/train/train_matched_baseline.py",
                                     "src/models/r3d_multitask.py",
                                     "src/data/echonet_dataset.py",
                                     "configs/base.yaml"])

    best_val_mae = float("inf")
    patience_counter = 0
    warmup_epochs = cfg["train"]["warmup_epochs"]

    for epoch in range(cfg["train"]["epochs"]):
        if epoch == warmup_epochs:
            print(f"[epoch {epoch+1}] Unfreezing backbone, switching to full fine-tune.")
            for p in model.backbone.parameters():
                p.requires_grad = True

        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False)

        print(
            f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.3f} | "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.3f}"
        )
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_mae": train_mae,
            "val_loss": val_loss, "val_mae": val_mae,
            "backbone_frozen": epoch < warmup_epochs,
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            ckpt_path = run_dir / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_mae": val_mae,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> New best val_mae={val_mae:.3f}. Saved to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{cfg['train']['early_stop_patience']})")
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.3f}")
    print(f"Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
