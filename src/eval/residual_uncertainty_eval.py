"""
Evaluates the residual-uncertainty model: the frozen mean predictor +
separately trained logvar head. Same conformal calibration and
correlation analysis as before, now on a model that (unlike the two
previous attempts) actually produces per-sample uncertainty.

Usage:
    python -m src.eval.residual_uncertainty_eval --config configs/base.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_residual_uncertainty import R3DResidualUncertainty

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/r3d18_residual_uncertainty_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_residual_uncertainty_best.pt")
    alpha = cfg["uncertainty"]["conformal_alpha"]

    out_dir = Path("results/residual_uncertainty")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mean_model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    mean_ckpt = torch.load(ckpt["mean_checkpoint"], map_location=device, weights_only=False)
    mean_model.load_state_dict(mean_ckpt["model_state_dict"])

    model = R3DResidualUncertainty(mean_model).to(device)
    model.logvar_head.load_state_dict(ckpt["logvar_head_state_dict"])
    model.eval()
    print(f"Loaded logvar head from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.3f}, "
          f"mean model val_mae={mean_ckpt['val_mae']:.3f}")

    def get_loader(split):
        ds = EchoNetDataset(
            cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split=split,
            num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
            ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
        )
        return ds, DataLoader(ds, batch_size=cfg["train"]["batch_size"] * 2, shuffle=False, num_workers=2)

    cal_ds, cal_loader = get_loader("VAL")
    test_ds, test_loader = get_loader("TEST")

    def predict(loader, n):
        mean_pred = np.zeros(n); sigma_pred = np.zeros(n); true = np.zeros(n)
        filenames = []
        i = 0
        with torch.no_grad():
            for batch in loader:
                clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
                bsz = clip.shape[0]
                out = model(clip)
                sigma = torch.exp(0.5 * out["ef_logvar"])
                mean_pred[i:i+bsz] = out["ef"].cpu().numpy()
                sigma_pred[i:i+bsz] = sigma.cpu().numpy()
                true[i:i+bsz] = batch["ef"].numpy()
                filenames.extend(batch["filename"])
                i += bsz
        return mean_pred, sigma_pred, true, filenames

    print("Running calibration set (VAL)...")
    cal_mean, cal_sigma, cal_true, _ = predict(cal_loader, len(cal_ds))
    print("Running test set (TEST)...")
    test_mean, test_sigma, test_true, test_files = predict(test_loader, len(test_ds))

    test_mae = np.abs(test_mean - test_true).mean()
    print(f"\nTest MAE (same frozen mean model as before): {test_mae:.3f}")
    print(f"Test sigma range: min={test_sigma.min():.2f}, mean={test_sigma.mean():.2f}, "
          f"max={test_sigma.max():.2f}, std={test_sigma.std():.3f}")

    cal_scores = np.abs(cal_mean - cal_true) / np.clip(cal_sigma, 1e-3, None)
    n_cal = len(cal_scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat = np.quantile(cal_scores, q_level)

    interval_halfwidth = q_hat * test_sigma
    lower, upper = test_mean - interval_halfwidth, test_mean + interval_halfwidth
    covered = (test_true >= lower) & (test_true <= upper)
    empirical_coverage = covered.mean()
    print(f"\nConformal q_hat = {q_hat:.3f} | Empirical coverage: {empirical_coverage:.1%} (target {1-alpha:.0%})")
    print(f"Mean interval width: {(2*interval_halfwidth).mean():.2f} EF points")

    test_abs_error = np.abs(test_mean - test_true)
    corr, pval = stats.spearmanr(test_sigma, test_abs_error)
    print(f"\nSpearman corr(sigma, |error|): {corr:.3f} (p={pval:.2e})")

    order = np.argsort(-test_sigma)
    deferral_fracs = np.linspace(0, 0.5, 11)
    selective_mae = []
    for frac in deferral_fracs:
        n_defer = int(frac * len(test_sigma))
        keep_idx = order[n_defer:]
        selective_mae.append(test_abs_error[keep_idx].mean() if len(keep_idx) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deferral_fracs * 100, selective_mae, marker="o", color="#1E8E5A")
    ax.set_xlabel("% most-uncertain cases deferred")
    ax.set_ylabel("MAE on remaining cases")
    ax.set_title("Selective prediction (residual-uncertainty model)")
    fig.savefig(out_dir / "selective_prediction_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(test_sigma, test_abs_error, s=6, alpha=0.4, color="#1E8E5A")
    ax.set_xlabel("Predicted sigma")
    ax.set_ylabel("|Predicted - True EF|")
    ax.set_title(f"Uncertainty vs. error (Spearman r={corr:.3f})")
    fig.savefig(out_dir / "sigma_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(out_dir / "residual_uncertainty_predictions.npz",
              filenames=test_files, mean_pred=test_mean, sigma_pred=test_sigma, true_ef=test_true,
              lower=lower, upper=upper, covered=covered)

    results = {
        "test_mae": float(test_mae),
        "test_sigma_stats": {"min": float(test_sigma.min()), "mean": float(test_sigma.mean()),
                              "max": float(test_sigma.max()), "std": float(test_sigma.std())},
        "conformal_q_hat": float(q_hat), "empirical_coverage": float(empirical_coverage),
        "target_coverage": float(1 - alpha), "mean_interval_width": float((2 * interval_halfwidth).mean()),
        "spearman_sigma_error_corr": float(corr), "spearman_pvalue": float(pval),
        "selective_mae_by_deferral": {f"{int(f*100)}%": float(m) for f, m in zip(deferral_fracs, selective_mae)},
    }
    with open(out_dir / "residual_uncertainty_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()