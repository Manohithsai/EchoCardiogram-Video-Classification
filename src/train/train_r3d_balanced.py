"""
R3D-18 retrain addressing class imbalance via Class-Balanced Loss
(Effective Number of Samples, Cui et al. CVPR 2019) with Deferred
Reweighting (DRW): trains normally for the first N epochs, then
switches on class-balanced loss weighting for the remainder. This is
more stable than WeightedRandomSampler oversampling (which was tried
first and diverged -- val_mae stuck at 17+ due to the ~33/33/33
effective sampling mix being too far from the real 77%-Normal
population) because it keeps the natural per-epoch data distribution
intact and only reweights each sample's loss contribution.

Usage:
    python -m src.train.train_r3d_balanced --config configs/base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset, ef_to_class
from src.models.r3d_multitask import R3DMultiTask, ef_consistency_loss

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]

BETA = 0.999  # effective-number hyperparameter -- standard choice from Cui et al.
DRW_START_EPOCH = 15  # train normally until here, then switch on class-balanced weighting


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def compute_class_balanced_weights(class_counts: np.ndarray, beta: float) -> np.ndarray:
    """effective_num = (1 - beta^n) / (1 - beta); weight = 1/effective_num,
    normalized so weights sum to num_classes (keeps overall loss magnitude
    comparable to the unweighted case)."""
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    return weights


def get_ef_class_tensor(ef_batch: torch.Tensor, thresholds: list, device) -> torch.Tensor:
    classes = torch.zeros_like(ef_batch, dtype=torch.long)
    for t in thresholds:
        classes += (ef_batch >= t).long()
    return classes.to(device)


def run_epoch(model, loader, optimizer, scaler, cfg, device, train, class_weights_tensor=None):
    """class_weights_tensor: None -> unweighted loss (pre-DRW phase).
    Otherwise, per-sample EF-regression loss is scaled by its class weight."""
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    thresholds = cfg["data"]["ef_thresholds"]
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
            ef, edv, esv = batch["ef"].to(device), batch["edv"].to(device), batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                out = model(clip)

                if class_weights_tensor is not None:
                    sample_classes = get_ef_class_tensor(ef, thresholds, device)
                    sample_weights = class_weights_tensor[sample_classes]
                    per_sample_ef_loss = nn.functional.smooth_l1_loss(out["ef"], ef, reduction="none")
                    loss_ef = (per_sample_ef_loss * sample_weights).mean()
                else:
                    loss_ef = nn.functional.smooth_l1_loss(out["ef"], ef)

                loss_edv = nn.functional.smooth_l1_loss(out["edv"], edv)
                loss_esv = nn.functional.smooth_l1_loss(out["esv"], esv)
                loss = loss_ef + loss_edv + loss_esv
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

    wandb.init(project="echo-ef", config=cfg, name="r3d18-class-balanced-drw")

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

    # Natural, unweighted DataLoader throughout -- the loss, not the
    # sampler, is where reweighting happens (and only after DRW_START_EPOCH).
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"], pin_memory=True)

    ef_classes = train_ds.df["EF"].apply(lambda ef: ef_to_class(ef, cfg["data"]["ef_thresholds"])).values
    class_counts = np.bincount(ef_classes, minlength=3)
    cb_weights = compute_class_balanced_weights(class_counts, BETA)
    print(f"Train class counts: Reduced={class_counts[0]}, Mildly reduced={class_counts[1]}, Normal={class_counts[2]}")
    print(f"Class-balanced loss weights (beta={BETA}): Reduced={cb_weights[0]:.3f}, "
          f"Mildly reduced={cb_weights[1]:.3f}, Normal={cb_weights[2]:.3f}")
    cb_weights_tensor = torch.tensor(cb_weights, dtype=torch.float32, device=device)

    model = R3DMultiTask(pretrained=True, dropout_p=cfg["model"]["dropout_p"]).to(device)

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

        use_cb_loss = epoch >= DRW_START_EPOCH
        if epoch == DRW_START_EPOCH:
            print(f"[epoch {epoch+1}] Switching on class-balanced loss weighting (DRW).")

        weights_arg = cb_weights_tensor if use_cb_loss else None
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True, class_weights_tensor=weights_arg)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False, class_weights_tensor=weights_arg)

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
              f"train_mae={train_mae:.3f} | val_mae={val_mae:.3f} | CB_loss={'ON' if use_cb_loss else 'off'}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_mae": train_mae,
                    "val_loss": val_loss, "val_mae": val_mae, "cb_loss_active": use_cb_loss})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            ckpt_path = ckpt_dir / "r3d18_balanced_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_mae": val_mae, "config": cfg, "cb_weights": cb_weights.tolist()}, ckpt_path)
            print(f"  -> New best val_mae={val_mae:.3f}. Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nDone. Best class-balanced val MAE: {best_val_mae:.3f}")


if __name__ == "__main__":
    main()