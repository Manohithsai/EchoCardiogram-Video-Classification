"""
Test-set evaluation for the frame-CNN baseline -- same metrics as
evaluate_test.py (used for R3D-18), so the two models can be compared
directly on identical test-set numbers.

Usage:
    python -m src.eval.evaluate_test_frame_cnn --config configs/base.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset, ef_to_class
from src.models.frame_cnn import FrameCNNMultiTask

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def bootstrap_ci(values, fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    stats_boot = [fn(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/frame_cnn_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "frame_cnn_best.pt")
    out_dir = Path("results/test_eval_frame_cnn")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FrameCNNMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_mae={checkpoint['val_mae']:.3f}")

    test_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="TEST",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} clips")

    all_pred_ef, all_true_ef = [], []
    all_pred_edv, all_true_edv = [], []
    all_pred_esv, all_true_esv = [], []
    all_filenames = []

    with torch.no_grad():
        for batch in test_loader:
            clip = normalize(batch["clip"].to(device), IMAGENET_MEAN, IMAGENET_STD, device)
            out = model(clip)
            all_pred_ef.append(out["ef"].cpu().numpy())
            all_true_ef.append(batch["ef"].numpy())
            all_pred_edv.append(out["edv"].cpu().numpy())
            all_true_edv.append(batch["edv"].numpy())
            all_pred_esv.append(out["esv"].cpu().numpy())
            all_true_esv.append(batch["esv"].numpy())
            all_filenames.extend(batch["filename"])

    pred_ef = np.concatenate(all_pred_ef)
    true_ef = np.concatenate(all_true_ef)
    pred_edv = np.concatenate(all_pred_edv)
    true_edv = np.concatenate(all_true_edv)
    pred_esv = np.concatenate(all_pred_esv)
    true_esv = np.concatenate(all_true_esv)

    np.savez(out_dir / "test_predictions.npz",
              filenames=all_filenames, pred_ef=pred_ef, true_ef=true_ef,
              pred_edv=pred_edv, true_edv=true_edv, pred_esv=pred_esv, true_esv=true_esv)

    results = {}
    errors = pred_ef - true_ef
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    pearson_r = stats.pearsonr(pred_ef, true_ef)[0]
    r2 = pearson_r ** 2

    idx_arr = np.arange(len(pred_ef))
    mae_ci = bootstrap_ci(idx_arr, lambda i: np.abs(pred_ef[i] - true_ef[i]).mean())
    rmse_ci = bootstrap_ci(idx_arr, lambda i: np.sqrt(((pred_ef[i] - true_ef[i]) ** 2).mean()))

    results["regression"] = {
        "mae": float(mae), "mae_95ci": [float(mae_ci[0]), float(mae_ci[1])],
        "rmse": float(rmse), "rmse_95ci": [float(rmse_ci[0]), float(rmse_ci[1])],
        "r2": float(r2), "pearson_r": float(pearson_r),
    }
    print(f"\n=== Regression (EF) -- Frame-CNN ===")
    print(f"MAE:  {mae:.3f}  (95% CI: {mae_ci[0]:.3f}-{mae_ci[1]:.3f})")
    print(f"RMSE: {rmse:.3f}  (95% CI: {rmse_ci[0]:.3f}-{rmse_ci[1]:.3f})")
    print(f"R²:   {r2:.3f}   Pearson r: {pearson_r:.3f}")

    for name, pred, true in [("EDV", pred_edv, true_edv), ("ESV", pred_esv, true_esv)]:
        mae_v = np.abs(pred - true).mean()
        results["regression"][f"{name.lower()}_mae"] = float(mae_v)
        print(f"{name} MAE: {mae_v:.3f}")

    thresholds = cfg["data"]["ef_thresholds"]
    pred_class = np.array([ef_to_class(v, thresholds) for v in pred_ef])
    true_class = np.array([ef_to_class(v, thresholds) for v in true_ef])
    class_names = ["Reduced", "Mildly reduced", "Normal"]

    pred_binary_50 = (pred_ef < 50).astype(int)
    true_binary_50 = (true_ef < 50).astype(int)
    auc_50 = roc_auc_score(true_binary_50, -pred_ef)
    tn, fp, fn, tp = confusion_matrix(true_binary_50, pred_binary_50).ravel()
    sens_50 = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec_50 = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    pred_binary_40 = (pred_ef < 40).astype(int)
    true_binary_40 = (true_ef < 40).astype(int)
    auc_40 = roc_auc_score(true_binary_40, -pred_ef)
    tn2, fp2, fn2, tp2 = confusion_matrix(true_binary_40, pred_binary_40).ravel()
    sens_40 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else float("nan")
    spec_40 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else float("nan")

    cm = confusion_matrix(true_class, pred_class)
    macro_f1_parts = []
    for c in range(3):
        tp_c = cm[c, c]
        fp_c = cm[:, c].sum() - tp_c
        fn_c = cm[c, :].sum() - tp_c
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_f1_parts.append(f1)
    macro_f1 = float(np.mean(macro_f1_parts))

    results["classification"] = {
        "auc_ef_below_50": float(auc_50), "sensitivity_below_50": float(sens_50), "specificity_below_50": float(spec_50),
        "auc_ef_below_40": float(auc_40), "sensitivity_below_40": float(sens_40), "specificity_below_40": float(spec_40),
        "macro_f1_3class": macro_f1, "confusion_matrix_3class": cm.tolist(), "class_names": class_names,
    }
    print(f"\n=== Classification -- Frame-CNN ===")
    print(f"EF<50 detection: AUC={auc_50:.3f}, Sens={sens_50:.3f}, Spec={spec_50:.3f}")
    print(f"EF<40 detection: AUC={auc_40:.3f}, Sens={sens_40:.3f}, Spec={spec_40:.3f}")
    print(f"3-class macro F1: {macro_f1:.3f}")
    print(f"Confusion matrix ({class_names}):\n{cm}")

    mean_vals = (pred_ef + true_ef) / 2
    diff_vals = pred_ef - true_ef
    bias = diff_vals.mean()
    sd = diff_vals.std()
    loa_upper, loa_lower = bias + 1.96 * sd, bias - 1.96 * sd
    results["bland_altman"] = {"bias": float(bias), "loa_upper": float(loa_upper), "loa_lower": float(loa_lower)}
    print(f"\n=== Bland-Altman -- Frame-CNN ===")
    print(f"Bias: {bias:.3f}, Limits of agreement: [{loa_lower:.3f}, {loa_upper:.3f}]")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_ef, pred_ef, s=6, alpha=0.4, color="#14213D")
    ax.plot([0, 100], [0, 100], color="#C25A2E", linestyle="--")
    ax.set_xlabel("True EF")
    ax.set_ylabel("Predicted EF")
    ax.set_title(f"Frame-CNN test set: predicted vs true EF (R²={r2:.3f})")
    fig.savefig(out_dir / "pred_vs_true_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()