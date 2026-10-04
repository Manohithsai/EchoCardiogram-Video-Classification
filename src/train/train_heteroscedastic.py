"""
Trains the heteroscedastic R3D-18 head. Initializes the backbone from the
existing best checkpoint (weights are compatible; only the final output
layer's shape changed from 3 to 4 outputs, so that one layer is skipped
and reinitialized while everything else transfers).

Usage:
    python -m src.train.train_heteroscedastic --config configs/base.yaml
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
from src.models.r3d_heteroscedastic import R3DHeteroscedastic, gaussian_nll_loss, ef_consistency_loss

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]

EPOCHS = 25
LR_HEAD = 5.0e-4
LR_BACKBONE = 1.0e-5
WARMUP_EPOCHS = 2  # short, since the backbone is already well-trained; mostly letting the new head settle


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def partial_load(model: R3DHeteroscedastic, old_state_dict: dict):
    """Loads every weight whose name AND shape match; skips the rest
    (the final output layer, which grew from 3 to 4 units) and reports
    exactly what was skipped so nothing silently fails."""
    model_dict = model.state_dict()
    matched, skipped = {}, []
    for k, v in old_state_dict.items():
        # Old model used self.head.{0,3} naming; new model uses self.shared.{0,2} + self.out.
        # Map the compatible layers explicitly.
        new_key = None
        if k.startswith("head.0."):
            new_key = k.replace("head.0.", "shared.0.")
        elif k == "backbone." or k.startswith("backbone."):
            new_key = k
        elif k.startswith("drop"):
            new_key = k  # parameter-free, nothing to load anyway

        if new_key and new_key in model_dict and model_dict[new_key].shape == v.shape:
            matched[new_key] = v
        else:
            skipped.append(k)

    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"Partial load: matched {len(matched)} tensors, skipped {len(skipped)} "
          f"(expected -- head.3.* doesn't exist in the new architecture's shape)")


def run_epoch(model, loader, optimizer, scaler, cfg, device, train):
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    total_loss, total_mae, n_batches = 0.0, 0.0, 0
    all_sigma = []

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
            ef, edv, esv = batch["ef"].to(device), batch["edv"].to(device), batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                out = model(clip)
                loss_ef = gaussian_nll_loss(out["ef"], out["ef_logvar"], ef)
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
                sigma = torch.exp(0.5 * out["ef_logvar"]).detach().cpu().numpy()
                all_sigma.append(sigma)

    sigma_arr = np.concatenate(all_sigma) if all_sigma else np.array([0.0])
    sigma_stats = {"min": sigma_arr.min(), "mean": sigma_arr.mean(), "max": sigma_arr.max(), "std": sigma_arr.std()}
    return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1), sigma_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/r3d18_mcdropout_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ckpt = args.init_checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_mcdropout_best.pt")

    wandb.init(project="echo-ef", config=cfg, name="r3d18-heteroscedastic")

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

    model = R3DHeteroscedastic(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    old_ckpt = torch.load(init_ckpt, map_location=device)
    partial_load(model, old_ckpt["model_state_dict"])
    print(f"Initialized from {init_ckpt} (val_mae was {old_ckpt['val_mae']:.3f})")

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.shared.parameters(), "lr": LR_HEAD},
            {"params": model.out.parameters(), "lr": LR_HEAD},
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"], init_scale=2.0**12)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    best_val_loss = float("inf")  # select on NLL now, not MAE -- MAE can wobble while variance learns
    patience_counter = 0
    patience = 10  # longer than before, since this model needs more epochs to differentiate sigma per-sample

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            print(f"[epoch {epoch+1}] Unfreezing backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

        train_loss, train_mae, train_sigma = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        val_loss, val_mae, val_sigma = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False)

        print(f"[Heteroscedastic] Epoch {epoch+1}/{EPOCHS} | "
              f"train_mae={train_mae:.3f} val_mae={val_mae:.3f} val_loss={val_loss:.3f} | "
              f"val_sigma[min={val_sigma['min']:.2f} mean={val_sigma['mean']:.2f} "
              f"max={val_sigma['max']:.2f} std={val_sigma['std']:.3f}]")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_mae": train_mae,
                    "val_loss": val_loss, "val_mae": val_mae, "val_sigma_std": val_sigma["std"]})

        # val_sigma std near 0 means every prediction is getting the same
        # uncertainty estimate -- exactly the collapse we saw before. Flag it
        # loudly instead of silently saving a useless checkpoint.
        if val_sigma["std"] < 0.05 and epoch > WARMUP_EPOCHS + 2:
            print(f"  [WARNING] val_sigma std is very low ({val_sigma['std']:.4f}) -- "
                  f"variance may be collapsing to a constant. Keep watching.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / "r3d18_heteroscedastic_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_mae": val_mae, "val_loss": val_loss, "val_sigma_std": val_sigma["std"],
                        "config": cfg}, ckpt_path)
            print(f"  -> New best val_loss={val_loss:.3f} (val_mae={val_mae:.3f}). Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nDone. Best heteroscedastic val_loss (NLL): {best_val_loss:.3f}")

if __name__ == "__main__":
    main()