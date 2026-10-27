"""
Re-evaluate ONE R3D-18 checkpoint on the TEST split and save everything needed
for provenance and correct metrics, into its own folder:

    results/eval_runs/<tag>/predictions.npz   (y_true, y_pred, edv/esv true+pred, file names if available)
    results/eval_runs/<tag>/metrics.json      (metrics + checkpoint name/hash/epoch + prediction hash)

Metrics reported:
  * true R2 = 1 - SSE/SST   AND   squared Pearson r2  (kept separate and labelled)
  * test variance SST/n  (must be IDENTICAL for every model on the same test set)
  * MAE / RMSE with bootstrap CIs, EDV/ESV MAE
  * EF<50 and EF<40 AUC / sensitivity / specificity, 3-class confusion matrix, per-class recall + MAE
  * Bland-Altman bias and limits of agreement

No training, no changes to checkpoints. Run from echo-ef-project/ :

    python -m src.eval.evaluate_all --config configs/base.yaml --checkpoint checkpoints\\r3d18_best.pt --tag r3d18_baseline
    python -m src.eval.evaluate_all --config configs/base.yaml --checkpoint checkpoints\\r3d18_balanced_best.pt --tag r3d18_cb_drw
    python -m src.eval.evaluate_all --config configs/base.yaml --checkpoint checkpoints\\r3d18_lds_best.pt --tag r3d18_lds
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]
CLASS_NAMES = ["Reduced", "Mildly reduced", "Normal"]


def sha256_file(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16] if n is None else h.hexdigest()[:n]


def normalize(clip, device, mean, std):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def to_class(ef, thresholds):
    cls = np.zeros(len(ef), dtype=int)
    for t in thresholds:
        cls += (ef >= t).astype(int)
    return cls


def boot_ci(values_fn, n, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [values_fn(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_score(y_bool, score):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_bool, score))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="unique folder name, e.g. r3d18_baseline")
    ap.add_argument("--arch", default="r3d18", choices=["r3d18", "swin3d_t"],
                    help="model architecture of the checkpoint (default r3d18)")
    ap.add_argument("--batch-size", type=int, default=None, help="override eval batch size (Swin: use 4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    thresholds = cfg["data"]["ef_thresholds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("results") / "eval_runs" / args.tag
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already has files. Choose a new --tag (never overwrite earlier results).")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint {args.checkpoint}: epoch {ckpt.get('epoch')}, val_mae={ckpt.get('val_mae')}")

    ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="TEST",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=thresholds, augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )
    loader = DataLoader(ds, batch_size=args.batch_size or cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    print(f"Test set: {len(ds)} clips")

    if args.arch == "swin3d_t":
        from src.models.swin3d_multitask import SwinMultiTask, swin_norm_stats
        model = SwinMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
        norm_mean, norm_std = ckpt.get("norm_mean") or swin_norm_stats()[0], ckpt.get("norm_std") or swin_norm_stats()[1]
    else:
        model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
        norm_mean, norm_std = R3D_MEAN, R3D_STD
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"arch={args.arch} normalisation mean={norm_mean} std={norm_std}")

    yt, yp, edv_t, edv_p, esv_t, esv_p = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            clip = normalize(batch["clip"].to(device), device, norm_mean, norm_std)
            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"] and device.type == "cuda"):
                out = model(clip)
            yt.append(batch["ef"].float().numpy()); yp.append(out["ef"].float().cpu().numpy())
            edv_t.append(batch["edv"].float().numpy()); edv_p.append(out["edv"].float().cpu().numpy())
            esv_t.append(batch["esv"].float().numpy()); esv_p.append(out["esv"].float().cpu().numpy())
    y_true, y_pred = np.concatenate(yt), np.concatenate(yp)
    edv_true, edv_pred = np.concatenate(edv_t), np.concatenate(edv_p)
    esv_true, esv_pred = np.concatenate(esv_t), np.concatenate(esv_p)
    n = len(y_true)

    names = None
    for col in ("FileName", "filename", "file_name"):
        if hasattr(ds, "df") and col in ds.df.columns:
            names = ds.df[col].astype(str).values[:n]
            break

    pred_path = out_dir / "predictions.npz"
    save = dict(y_true=y_true, y_pred=y_pred, edv_true=edv_true, edv_pred=edv_pred,
                esv_true=esv_true, esv_pred=esv_pred)
    if names is not None:
        save["file_names"] = names
    np.savez(pred_path, **save)

    err = y_pred - y_true
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae_ci = boot_ci(lambda i: float(np.mean(np.abs(err[i]))), n)
    rmse_ci = boot_ci(lambda i: float(np.sqrt(np.mean(err[i] ** 2))), n)

    cls_true, cls_pred = to_class(y_true, thresholds), to_class(y_pred, thresholds)
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(cls_true, cls_pred):
        cm[a, b] += 1
    per_class_mae = [float(np.mean(np.abs(err[cls_true == c]))) for c in range(3)]
    per_class_recall = [float(cm[c, c] / cm[c].sum()) for c in range(3)]
    f1s = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)

    def detect(th):
        pos = y_true < th
        pred_pos = y_pred < th
        sens = float((pred_pos & pos).sum() / pos.sum())
        spec = float((~pred_pos & ~pos).sum() / (~pos).sum())
        return {"auc": auc_score(pos, -y_pred), "sensitivity": sens, "specificity": spec}

    bias = float(err.mean())
    sd = float(err.std(ddof=1))
    metrics = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "checkpoint_sha256_16": sha256_file(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_mae": ckpt.get("val_mae"),
        "predictions_file": str(pred_path),
        "predictions_sha256_16": sha256_file(pred_path),
        "n_test": int(n),
        "mae": mae, "mae_ci95": mae_ci, "rmse": rmse, "rmse_ci95": rmse_ci,
        "r2_coefficient_of_determination": 1.0 - sse / sst,
        "pearson_r": r, "pearson_r_squared": r ** 2,
        "test_variance_sst_over_n": sst / n,
        "edv_mae": float(np.mean(np.abs(edv_pred - edv_true))),
        "esv_mae": float(np.mean(np.abs(esv_pred - esv_true))),
        "ef_lt_50": detect(50), "ef_lt_40": detect(40),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "class_names": CLASS_NAMES,
        "per_class_mae": dict(zip(CLASS_NAMES, per_class_mae)),
        "per_class_recall": dict(zip(CLASS_NAMES, per_class_recall)),
        "bland_altman": {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd},
    }
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)

    print(f"\n=== {args.tag} (n={n}) ===")
    print(f"MAE {mae:.3f} (95% CI {mae_ci[0]:.3f}-{mae_ci[1]:.3f})   RMSE {rmse:.3f} (95% CI {rmse_ci[0]:.3f}-{rmse_ci[1]:.3f})")
    print(f"R2 (true, 1-SSE/SST) = {metrics['r2_coefficient_of_determination']:.3f}   "
          f"Pearson r = {r:.3f}   r^2 = {r**2:.3f}")
    print(f"Test variance SST/n = {sst/n:.2f}   <- must match across all models")
    print(f"RMSE^2/(1-R2_true)  = {rmse**2/(1-metrics['r2_coefficient_of_determination']):.2f}")
    print(f"EDV MAE {metrics['edv_mae']:.3f}   ESV MAE {metrics['esv_mae']:.3f}")
    for k in ("ef_lt_50", "ef_lt_40"):
        d = metrics[k]
        auc = "n/a" if d["auc"] is None else f"{d['auc']:.3f}"
        print(f"{k}: AUC {auc}  sens {d['sensitivity']:.3f}  spec {d['specificity']:.3f}")
    print(f"3-class macro F1 {metrics['macro_f1']:.3f}")
    print("Confusion matrix (rows true, cols pred):", CLASS_NAMES)
    print(cm)
    print("Per-class recall:", {k: round(v, 3) for k, v in metrics["per_class_recall"].items()})
    print("Per-class MAE:   ", {k: round(v, 2) for k, v in metrics["per_class_mae"].items()})
    print(f"Bland-Altman bias {bias:.3f}  LoA [{bias-1.96*sd:.3f}, {bias+1.96*sd:.3f}]")
    print(f"checkpoint sha256[:16] = {metrics['checkpoint_sha256_16']}   predictions sha256[:16] = {metrics['predictions_sha256_16']}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()