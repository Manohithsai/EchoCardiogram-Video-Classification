"""
Train Video Swin (Swin3D-T) for EF / EDV / ESV with the SAME recipe as the R3D-18 baseline:
unweighted smooth-L1, EF-consistency loss from the config, frozen-backbone warm-up then
unfreeze, checkpoint = lowest OVERALL validation MAE, early stopping on validation.
(So the R3D-18 vs Video Swin comparison differs only in the architecture.)

Run from echo-ef-project/ :

  1) memory / speed check (3 batches, then exits):
     python -m src.train.train_swin3d --config configs/base.yaml --dry-run --batch-size 4

  2) full training (use the batch size that fit in step 1; --grad-accum keeps the
     effective batch size = batch-size * grad-accum):
     python -m src.train.train_swin3d --config configs/base.yaml --batch-size 4 --grad-accum 2
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import ef_consistency_loss
from src.models.swin3d_multitask import SwinMultiTask, swin_norm_stats

CKPT_NAME = "swin3d_t_best.pt"
RUN_NAME = "swin3d-t"


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def to_class(ef_np, thresholds):
    cls = np.zeros(len(ef_np), dtype=int)
    for t in thresholds:
        cls += (ef_np >= t).astype(int)
    return cls


def run_epoch(model, loader, optimizer, scaler, cfg, device, mean, std, train, accum=1, max_batches=None,
              timings=None):
    model.train() if train else model.eval()
    cw = cfg["model"]["ef_consistency_weight"]
    thresholds = cfg["data"]["ef_thresholds"]
    total_loss, n_batches = 0.0, 0
    preds, trues = [], []
    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            clip = normalize(batch["clip"].to(device), mean, std, device)
            ef, edv, esv = batch["ef"].to(device), batch["edv"].to(device), batch["esv"].to(device)

            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"] and device.type == "cuda"):
                out = model(clip)
                loss = (nn.functional.smooth_l1_loss(out["ef"], ef)
                        + nn.functional.smooth_l1_loss(out["edv"], edv)
                        + nn.functional.smooth_l1_loss(out["esv"], esv))
                if cw > 0:
                    loss = loss + cw * ef_consistency_loss(out["ef"], out["edv"], out["esv"])

            if train:
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            lv = loss.item()
            if timings is not None:  # loss.item() already synchronised the GPU
                timings.append(time.time())
            if np.isfinite(lv):
                total_loss += lv
                n_batches += 1
                preds.append(out["ef"].detach().float().cpu().numpy())
                trues.append(ef.detach().float().cpu().numpy())

    pred, true = np.concatenate(preds), np.concatenate(trues)
    err = np.abs(pred - true)
    cls = to_class(true, thresholds)
    per_class = [float(err[cls == c].mean()) if (cls == c).any() else float("nan") for c in range(len(thresholds) + 1)]
    return {"loss": total_loss / max(n_batches, 1), "mae": float(err.mean()),
            "per_class_mae": per_class, "class_avg_mae": float(np.nanmean(per_class))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=None, help="override cfg batch size (6GB GPU: try 4 or 2)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-backbone", type=float, default=None, help="override backbone LR (Swin usually wants <= 1e-4)")
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="run a short benchmark, print GPU memory/time, exit")
    ap.add_argument("--bench-batches", type=int, default=24, help="batches to time in --dry-run (first 4 are ignored)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bs = args.batch_size or cfg["train"]["batch_size"]
    lr_bb = args.lr_backbone or cfg["train"]["lr_backbone"]
    lr_head = args.lr_head or cfg["train"]["lr_head"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = swin_norm_stats()
    print(f"device={device} batch_size={bs} grad_accum={args.grad_accum} lr_backbone={lr_bb} lr_head={lr_head}")
    print(f"normalisation mean={mean} std={std}")

    def make_ds(split, aug):
        return EchoNetDataset(
            cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split=split,
            num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
            ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=aug), seed=cfg["seed"])

    train_ds, val_ds = make_ds("TRAIN", True), make_ds("VAL", False)
    print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg["train"]["num_workers"],
                            pin_memory=True)

    model = SwinMultiTask(pretrained=True, dropout_p=cfg["model"]["dropout_p"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SwinMultiTask parameters: {n_params:.1f}M")

    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": lr_head},
         {"params": model.backbone.parameters(), "lr": lr_bb}],
        weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda", init_scale=2.0 ** 12)

    if args.dry_run:
        for p in model.backbone.parameters():
            p.requires_grad = True  # worst case for memory
        stamps = [time.time()]
        run_epoch(model, train_loader, optimizer, scaler, cfg, device, mean, std, True, args.grad_accum,
                  max_batches=args.bench_batches, timings=stamps)
        gaps = np.diff(stamps)  # seconds per batch, includes data loading
        steady = gaps[4:] if len(gaps) > 6 else gaps  # skip start-up batches (CUDA init, worker spawn)
        med, mean_gap = float(np.median(steady)), float(np.mean(steady))
        if device.type == "cuda":
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (of ~6 GB)")
        print(f"first 4 batches (start-up): {', '.join(f'{g:.1f}s' for g in gaps[:4])}")
        print(f"steady state over {len(steady)} batches: median {med:.2f} s, mean {mean_gap:.2f} s per batch of {bs}")
        n_b = len(train_loader)
        print(f"-> ~{med * n_b / 60:.0f} min per epoch (train only, median) / ~{mean_gap * n_b / 60:.0f} min (mean); "
              f"validation adds roughly 15-20% more")
        print("Dry run OK. If memory is close to 6 GB, lower --batch-size and raise --grad-accum.")
        return

    wandb.init(project="echo-ef", name=RUN_NAME,
               config={**cfg, "arch": "swin3d_t", "batch_size": bs, "grad_accum": args.grad_accum,
                       "lr_backbone": lr_bb, "lr_head": lr_head})
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, patience = float("inf"), 0
    warmup = cfg["train"]["warmup_epochs"]

    for epoch in range(cfg["train"]["epochs"]):
        if epoch == warmup:
            print(f"[epoch {epoch + 1}] Unfreezing backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True
        tr = run_epoch(model, train_loader, optimizer, scaler, cfg, device, mean, std, True, args.grad_accum)
        va = run_epoch(model, val_loader, optimizer, scaler, cfg, device, mean, std, False)
        pc = va["per_class_mae"]
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']} | train_mae={tr['mae']:.3f} | val_mae={va['mae']:.3f} | "
              f"val_class_avg_mae={va['class_avg_mae']:.3f} | per-class R/M/N = " + "/".join(f"{x:.2f}" for x in pc))
        log = {"epoch": epoch + 1, "train_loss": tr["loss"], "train_mae": tr["mae"], "val_loss": va["loss"],
               "val_mae": va["mae"], "val_class_avg_mae": va["class_avg_mae"],
               "backbone_frozen": epoch < warmup}
        for i, x in enumerate(pc):
            log[f"val_mae_class{i}"] = x
        wandb.log(log)

        if va["mae"] < best:  # same selection rule as the R3D-18 baseline: overall val MAE
            best, patience = va["mae"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_mae": va["mae"],
                        "val_class_avg_mae": va["class_avg_mae"], "arch": "swin3d_t", "config": cfg,
                        "norm_mean": mean, "norm_std": std}, ckpt_dir / CKPT_NAME)
            print(f"  -> New best val_mae={best:.3f}. Saved to {ckpt_dir / CKPT_NAME}")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    wandb.finish()
    print(f"\nDone. Best overall val MAE: {best:.3f}")


if __name__ == "__main__":
    main()