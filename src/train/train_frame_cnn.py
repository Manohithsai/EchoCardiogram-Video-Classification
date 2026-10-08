"""
Trains the frame-level CNN baseline (ResNet-18 + temporal pooling).
Tests whether full 3D video modeling (R3D) actually beats a much
simpler per-frame approach -- the key comparison this baseline exists for.

Usage:
    python -m src.train.train_frame_cnn --config configs/base.yaml
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
from src.models.frame_cnn import FrameCNNMultiTask

# ResNet-18 was pretrained on ImageNet with these stats -- different from
# R3D-18's Kinetics stats. Using the wrong ones silently hurts transfer.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def ef_consistency_loss(pred_ef, pred_edv, pred_esv):
    eps = 1e-4
    implied_ef = (pred_edv - pred_esv) / (pred_edv.clamp(min=eps)) * 100.0
    return torch.abs(pred_ef - implied_ef).mean()


def run_epoch(model, loader, optimizer, scaler, cfg, device, train):
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), IMAGENET_MEAN, IMAGENET_STD, device)
            ef, edv, esv = batch["ef"].to(device), batch["edv"].to(device), batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                out = model(clip)
                loss = (nn.functional.smooth_l1_loss(out["ef"], ef)
                        + nn.functional.smooth_l1_loss(out["edv"], edv)
                        + nn.functional.smooth_l1_loss(out["esv"], esv))
                if consistency_weight > 0:
                    loss = loss + consistency_weight * ef_consistency_loss(out["ef"], out["edv"], out["esv"])

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

    return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="echo-ef", config=cfg, name="frame-cnn-baseline")

    train_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="TRAIN",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=True), seed=cfg["seed"],
    )
    val_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="VAL",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )
    print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = FrameCNNMultiTask(pretrained=True, dropout_p=cfg["model"]["dropout_p"]).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": cfg["train"]["lr_head"]},
            {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"], init_scale=2.0**12)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    patience_counter = 0
    warmup_epochs = cfg["train"]["warmup_epochs"]

    for epoch in range(cfg["train"]["epochs"]):
        if epoch == warmup_epochs:
            print(f"[epoch {epoch+1}] Unfreezing backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False)

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
              f"train_mae={train_mae:.3f} | val_mae={val_mae:.3f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_mae": train_mae,
                    "val_loss": val_loss, "val_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            ckpt_path = ckpt_dir / "frame_cnn_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_mae": val_mae, "config": cfg}, ckpt_path)
            print(f"  -> New best val_mae={val_mae:.3f}. Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.3f}")
    print(f"Best checkpoint: {ckpt_dir / 'frame_cnn_best.pt'}")


if __name__ == "__main__":
    main()