"""
Re-evaluation of ONE saved checkpoint on the TEST split, written to a NEW folder.

Inference and preprocessing are copied from src/eval/evaluate_test_frame_cnn.py (same model
class, same dataset arguments, same normalisation, batch size and loader order).
Nothing is trained and no existing file is touched or overwritten.

Differences from evaluate_test_frame_cnn.py:
  * --out-dir is required and must NOT exist yet (the run stops if it does).
  * Checkpoint is read with torch.load(..., weights_only=True) on CPU, then the
    model is moved to the device. No unsafe-pickle fallback.
  * Reports the true R2 (1 - SSE/SST) AND the squared Pearson correlation.
  * Saves provenance (checkpoint path + SHA256, stored checkpoint config, config
    file SHA256, split, clip count, library versions) next to the results.

Usage (from the project root, venv active):
    python -m src.eval.recheck_frame_cnn --config configs/base.yaml ^
        --checkpoint checkpoints/frame_cnn_best.pt --out-dir results_recheck/frame_cnn
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def regression_metrics(pred, true):
    """MAE, RMSE, true R2 (1 - SSE/SST), squared Pearson, Pearson r."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 1 or len(pred) < 3:
        raise ValueError(f"bad shapes: pred {pred.shape}, true {true.shape}")
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        raise ValueError("non-finite values in predictions or targets")
    err = pred - true
    sst = float(((true - true.mean()) ** 2).sum())
    r = float(stats.pearsonr(pred, true)[0])
    return {
        "n": int(len(pred)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2_true": float(1.0 - (err ** 2).sum() / sst),
        "pearson_r": r,
        "pearson_r_squared": r ** 2,
        "mean_error_bias": float(err.mean()),
    }


def main():
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
    from src.models.frame_cnn import FrameCNNMultiTask

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True,
                        help="NEW folder; the run stops if it already exists")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)  # refuses an existing folder
    except FileExistsError:
        raise SystemExit(f"Refusing to run: {out_dir} already exists. Pick a new folder name.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint safely on CPU, then move the model to the device ---
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise SystemExit("Checkpoint is not a dict with 'model_state_dict'; stopping.")
    model = FrameCNNMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded {ckpt_path.name}: epoch={checkpoint.get('epoch')}, val_mae={checkpoint.get('val_mae')}")

    # --- Test data (no augmentation), same arguments as evaluate_test_frame_cnn.py ---
    test_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"],
        file_list_csv=cfg["data"]["file_list_csv"],
        split="TEST",
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"],
        augment=AugmentConfig(temporal=False),
        seed=cfg["seed"],
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} clips")

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)

    cols = {k: [] for k in ["pred_ef", "true_ef", "pred_edv", "true_edv", "pred_esv", "true_esv"]}
    filenames = []
    with torch.no_grad():
        for batch in loader:
            clip = (batch["clip"].to(device) - mean_t) / std_t
            out = model(clip)
            cols["pred_ef"].append(out["ef"].cpu().numpy())
            cols["true_ef"].append(batch["ef"].numpy())
            cols["pred_edv"].append(out["edv"].cpu().numpy())
            cols["true_edv"].append(batch["edv"].numpy())
            cols["pred_esv"].append(out["esv"].cpu().numpy())
            cols["true_esv"].append(batch["esv"].numpy())
            filenames.extend(batch["filename"])
    arrays = {k: np.concatenate(v) for k, v in cols.items()}

    np.savez(out_dir / "test_predictions.npz", filenames=filenames, **arrays)

    metrics = regression_metrics(arrays["pred_ef"], arrays["true_ef"])
    metrics["edv_mae"] = float(np.abs(arrays["pred_edv"] - arrays["true_edv"]).mean())
    metrics["esv_mae"] = float(np.abs(arrays["pred_esv"] - arrays["true_esv"]).mean())

    provenance = {
        "created_local_time": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mae": checkpoint.get("val_mae"),
        "checkpoint_stored_config": checkpoint.get("config"),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "file_list_csv": cfg["data"]["file_list_csv"],
        "file_list_csv_sha256": sha256_file(cfg["data"]["file_list_csv"]),
        "model_class": "FrameCNNMultiTask",
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "split": "TEST",
        "n_clips": len(filenames),
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "note": ("Inference-only re-evaluation. The TEST split was used during earlier model "
                 "development, so this verifies saved models; it is not an untouched final test."),
    }
    with open(out_dir / "recheck_results.json", "w") as f:
        json.dump({"metrics": metrics, "provenance": provenance}, f, indent=2)

    print("\n=== EF regression (this checkpoint, TEST) ===")
    print(f"MAE        : {metrics['mae']:.3f}")
    print(f"RMSE       : {metrics['rmse']:.3f}")
    print(f"R2 (true)  : {metrics['r2_true']:.3f}")
    print(f"Pearson^2  : {metrics['pearson_r_squared']:.3f}")
    print(f"Saved to   : {out_dir}")


if __name__ == "__main__":
    main()
