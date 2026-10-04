"""
Short continued fine-tune after adding backbone dropout to R3DMultiTask.
Loads the existing best checkpoint's weights (compatible, since dropout
layers add no parameters), then fine-tunes briefly so the network adapts
to seeing stochastic feature maps during training -- consistent with how
MC Dropout is meant to work, rather than only turning dropout on at
inference with a model that never trained under it.

Usage:
    python -m src.train.finetune_with_backbone_dropout --config configs/base.yaml
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

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]

FINETUNE_EPOCHS = 12
FINETUNE_LR = 1.0e-5  # low, since the model is already well-trained -- this is adaptation, not learning from scratch


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def run_epoch(model, loader, optimizer, scaler, cfg, device, train):
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
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
    parser.add_argument("--init_checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/r3d18_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ckpt = args.init_checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_best.pt")

    wandb.init(project="echo-ef", config=cfg, name="r3d18-backbone-dropout-finetune")

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
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # New architecture (with backbone dropout), loaded from old weights.
    # pretrained=False because we're about to overwrite with our own trained weights.
    model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    old_ckpt = torch.load(init_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(old_ckpt["model_state_dict"], strict=True)
    print(f"Loaded weights from {init_ckpt} (epoch {old_ckpt['epoch']}, val_mae={old_ckpt['val_mae']:.3f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"], init_scale=2.0**12)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    best_val_mae = float("inf")

    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False)

        print(f"[Finetune] Epoch {epoch+1}/{FINETUNE_EPOCHS} | "
              f"train_mae={train_mae:.3f} | val_mae={val_mae:.3f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_mae": train_mae,
                    "val_loss": val_loss, "val_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            ckpt_path = ckpt_dir / "r3d18_mcdropout_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_mae": val_mae, "config": cfg}, ckpt_path)
            print(f"  -> New best val_mae={val_mae:.3f}. Saved to {ckpt_path}")

    wandb.finish()
    print(f"\nDone. Best val MAE with backbone dropout: {best_val_mae:.3f}")
    print(f"(Compare to original: {old_ckpt['val_mae']:.3f} -- a small regression here is expected and fine, "
          f"since you've traded a bit of raw accuracy for a model whose uncertainty should now be meaningful.)")


if __name__ == "__main__":
    main()