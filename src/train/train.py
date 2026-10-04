"""
Training loop for EchoNet-Dynamic EF estimation.

Supports: R3D-18 multi-task model (EF/EDV/ESV), mixed precision (AMP),
EF-consistency loss, wandb logging, checkpointing on best val MAE,
early stopping. Tuned defaults for a 6GB RTX 3060 (small batch size,
AMP on, grad clipping).

Usage:
    python -m src.train.train --config configs/base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask, ef_consistency_loss

# R3D-18 was pretrained on Kinetics-400 with these normalization stats.
# Using ImageNet stats here would silently hurt transfer learning.
R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]


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
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available -- training will be very slow on CPU.")

    wandb.init(project="echo-ef", config=cfg, name="r3d18-run")

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
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            ckpt_path = ckpt_dir / "r3d18_best.pt"
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
    print(f"Best checkpoint: {ckpt_dir / 'r3d18_best.pt'}")


if __name__ == "__main__":
    main()