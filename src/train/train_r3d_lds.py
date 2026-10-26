"""
R3D-18 retrain addressing class imbalance with Label Distribution Smoothing
(LDS) reweighting (Yang et al., "Delving into Deep Imbalanced Regression",
ICML 2021).

Instead of three coarse class weights, every training clip is weighted by the
inverse of the *smoothed* density of its EF value, so rare EF ranges (very low
EF) get large weights and the weights change smoothly across the 40/50
thresholds. Weighting is applied for the whole run (not deferred), only to the
EF regression loss, and only on the training split.

Checkpoint selection and early stopping use the CLASS-AVERAGED validation MAE
(mean of per-class MAE for Reduced / Mildly reduced / Normal), so the model
is not rewarded for only getting the 78% Normal class right.

Usage (from echo-ef-project/):
    python -m src.train.train_r3d_lds --config configs/base.yaml
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

# ---- LDS hyperparameters (tune these if needed) ----
LDS_KERNEL_SIZE = 5     # Gaussian kernel width in EF bins
LDS_SIGMA = 2.0         # Gaussian sigma in EF bins
LDS_POWER = 1.0         # weight = density^-power  (0.5 = softer, sqrt-inverse)
LDS_MAX_WEIGHT = 5.0    # clip so a handful of extreme-EF clips cannot dominate
EF_BINS = 101           # integer EF bins 0..100

CKPT_NAME = "r3d18_lds_best.pt"
RUN_NAME = "r3d18-lds"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def compute_lds_bin_weights(train_ef: np.ndarray) -> np.ndarray:
    """Return an array of length EF_BINS: weight for each integer EF bin.
    Weights are normalized so the mean weight over TRAIN SAMPLES is 1, which
    keeps the overall loss scale comparable to the unweighted run."""
    bins = np.clip(np.rint(train_ef).astype(int), 0, EF_BINS - 1)
    hist = np.bincount(bins, minlength=EF_BINS).astype(np.float64)

    half = LDS_KERNEL_SIZE // 2
    xs = np.arange(-half, half + 1)
    kernel = np.exp(-(xs ** 2) / (2 * LDS_SIGMA ** 2))
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")

    smoothed = np.maximum(smoothed, 1e-3)  # avoid division by ~0 in empty bins
    raw = smoothed ** (-LDS_POWER)

    def renorm(w):
        return w / (w[bins].mean())

    w = renorm(raw)
    w = np.minimum(w, LDS_MAX_WEIGHT)
    w = renorm(w)
    return w.astype(np.float32)


def ef_to_class_tensor(ef: torch.Tensor, thresholds) -> torch.Tensor:
    cls = torch.zeros_like(ef, dtype=torch.long)
    for t in thresholds:
        cls += (ef >= t).long()
    return cls


def run_epoch(model, loader, optimizer, scaler, cfg, device, train, lds_weights=None):
    """lds_weights: tensor of length EF_BINS (train only) or None.
    Returns dict with loss, overall MAE, per-class MAE and class-averaged MAE."""
    model.train() if train else model.eval()
    consistency_weight = cfg["model"]["ef_consistency_weight"]
    thresholds = cfg["data"]["ef_thresholds"]
    n_classes = len(thresholds) + 1
    total_loss, n_batches = 0.0, 0
    all_pred, all_true = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
            ef, edv, esv = batch["ef"].to(device), batch["edv"].to(device), batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                out = model(clip)

                if lds_weights is not None:
                    idx = ef.round().clamp(0, EF_BINS - 1).long()
                    sample_w = lds_weights[idx]
                    per_sample = nn.functional.smooth_l1_loss(out["ef"], ef, reduction="none")
                    loss_ef = (per_sample * sample_w).mean()
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
                n_batches += 1
                all_pred.append(out["ef"].detach().float().cpu())
                all_true.append(ef.detach().float().cpu())

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    abs_err = (pred - true).abs()
    cls = ef_to_class_tensor(true, thresholds)
    per_class = []
    for c in range(n_classes):
        m = cls == c
        per_class.append(abs_err[m].mean().item() if m.any() else float("nan"))
    class_avg = float(np.nanmean(per_class))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": abs_err.mean().item(),
        "per_class_mae": per_class,
        "class_avg_mae": class_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="echo-ef", config={**cfg, "lds_sigma": LDS_SIGMA, "lds_kernel": LDS_KERNEL_SIZE,
                                          "lds_power": LDS_POWER, "lds_max_weight": LDS_MAX_WEIGHT},
               name=RUN_NAME)

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

    train_ef = train_ds.df["EF"].values.astype(np.float64)
    bin_weights = compute_lds_bin_weights(train_ef)
    sample_bins = np.clip(np.rint(train_ef).astype(int), 0, EF_BINS - 1)
    sample_w = bin_weights[sample_bins]
    thresholds = cfg["data"]["ef_thresholds"]
    train_cls = np.zeros(len(train_ef), dtype=int)
    for t in thresholds:
        train_cls += (train_ef >= t).astype(int)
    print(f"LDS weights (kernel={LDS_KERNEL_SIZE}, sigma={LDS_SIGMA}, power={LDS_POWER}, max={LDS_MAX_WEIGHT})")
    print(f"  weight range over train clips: min={sample_w.min():.3f}, max={sample_w.max():.3f}, mean={sample_w.mean():.3f}")
    for c, name in enumerate(["Reduced", "Mildly reduced", "Normal"][: len(thresholds) + 1]):
        print(f"  mean weight, {name}: {sample_w[train_cls == c].mean():.3f}")
    lds_weights_tensor = torch.tensor(bin_weights, dtype=torch.float32, device=device)

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

    best_score = float("inf")  # class-averaged val MAE
    patience_counter = 0
    warmup_epochs = cfg["train"]["warmup_epochs"]

    for epoch in range(cfg["train"]["epochs"]):
        if epoch == warmup_epochs:
            print(f"[epoch {epoch+1}] Unfreezing backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

        tr = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True,
                       lds_weights=lds_weights_tensor)
        va = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False, lds_weights=None)

        pc = va["per_class_mae"]
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train_mae={tr['mae']:.3f} | "
              f"val_mae={va['mae']:.3f} | val_class_avg_mae={va['class_avg_mae']:.3f} | "
              f"val per-class MAE R/M/N = " + "/".join(f"{x:.2f}" for x in pc))
        log = {"epoch": epoch + 1, "train_loss": tr["loss"], "train_mae": tr["mae"],
               "val_loss": va["loss"], "val_mae": va["mae"], "val_class_avg_mae": va["class_avg_mae"]}
        for i, x in enumerate(pc):
            log[f"val_mae_class{i}"] = x
        wandb.log(log)

        if va["class_avg_mae"] < best_score:
            best_score = va["class_avg_mae"]
            patience_counter = 0
            ckpt_path = ckpt_dir / CKPT_NAME
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_mae": va["mae"], "val_class_avg_mae": va["class_avg_mae"],
                        "config": cfg, "lds_bin_weights": bin_weights.tolist()}, ckpt_path)
            print(f"  -> New best val_class_avg_mae={best_score:.3f} (overall val_mae={va['mae']:.3f}). Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nDone. Best class-averaged val MAE: {best_score:.3f}")


if __name__ == "__main__":
    main()