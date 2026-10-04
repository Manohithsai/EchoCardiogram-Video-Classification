"""
Trains ONLY the logvar head on top of a frozen, already-trained R3D-18.
Much faster than the joint approach -- no backward pass through the 3D
backbone at all, just a small MLP.

Usage:
    python -m src.train.train_residual_uncertainty --config configs/base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_residual_uncertainty import R3DResidualUncertainty, gaussian_nll_loss

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]

EPOCHS = 20
LR = 1.0e-3  # can afford to be higher -- it's a tiny MLP with no risk to pretrained features
PATIENCE = 6


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def run_epoch(model, loader, optimizer, cfg, device, train):
    model.logvar_head.train() if train else model.logvar_head.eval()
    total_loss, n_batches = 0.0, 0
    all_sigma = []

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
            ef = batch["ef"].to(device)

            out = model(clip)
            loss = gaussian_nll_loss(out["ef"], out["ef_logvar"], ef)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            if np.isfinite(loss_val):
                total_loss += loss_val
                n_batches += 1
                sigma = torch.exp(0.5 * out["ef_logvar"]).detach().cpu().numpy()
                all_sigma.append(sigma)

    sigma_arr = np.concatenate(all_sigma) if all_sigma else np.array([0.0])
    return (total_loss / max(n_batches, 1),
            {"min": sigma_arr.min(), "mean": sigma_arr.mean(), "max": sigma_arr.max(), "std": sigma_arr.std()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mean_checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/r3d18_mcdropout_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_ckpt_path = args.mean_checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_mcdropout_best.pt")

    wandb.init(project="echo-ef", config=cfg, name="r3d18-residual-uncertainty")

    train_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="TRAIN",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
        # Note: temporal augment OFF here even for train -- the frozen mean
        # model's errors should reflect clean inference conditions, not be
        # muddied by extra clip-sampling randomness.
    )
    val_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="VAL",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"] * 2, shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"] * 2, shuffle=False, num_workers=cfg["train"]["num_workers"])

    mean_model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    mean_ckpt = torch.load(mean_ckpt_path, map_location=device)
    mean_model.load_state_dict(mean_ckpt["model_state_dict"])
    print(f"Loaded frozen mean model from {mean_ckpt_path} (val_mae={mean_ckpt['val_mae']:.3f})")

    model = R3DResidualUncertainty(mean_model).to(device)
    optimizer = torch.optim.Adam(model.logvar_head.parameters(), lr=LR)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_sigma = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_loss, val_sigma = run_epoch(model, val_loader, optimizer, cfg, device, train=False)

        print(f"[ResidualUncertainty] Epoch {epoch+1}/{EPOCHS} | "
              f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} | "
              f"val_sigma[min={val_sigma['min']:.2f} mean={val_sigma['mean']:.2f} "
              f"max={val_sigma['max']:.2f} std={val_sigma['std']:.3f}]")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                    "val_sigma_std": val_sigma["std"]})

        if val_sigma["std"] < 0.05 and epoch > 3:
            print(f"  [WARNING] val_sigma std still near 0 -- check logvar_head is receiving gradients.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / "r3d18_residual_uncertainty_best.pt"
            torch.save({"logvar_head_state_dict": model.logvar_head.state_dict(),
                        "mean_checkpoint": mean_ckpt_path, "epoch": epoch + 1,
                        "val_loss": val_loss, "val_sigma_std": val_sigma["std"], "config": cfg}, ckpt_path)
            print(f"  -> New best val_loss={val_loss:.3f}. Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nDone. Best val_loss (NLL): {best_val_loss:.3f}")


if __name__ == "__main__":
    main()