"""
MC Dropout uncertainty estimation + conformal prediction, run on the
already-trained R3D-18 checkpoint. No retraining needed -- dropout is
kept active at inference time to get a distribution of predictions per
clip, which becomes the basis for calibrated intervals.

Usage:
    python -m src.eval.mc_dropout --config configs/base.yaml
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

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def mc_dropout_predict(model, clip, n_passes: int) -> np.ndarray:
    """Runs n_passes stochastic forward passes with dropout active.
    Returns array of shape (n_passes,) of EF predictions for this batch item,
    actually (n_passes, batch_size) -- caller reshapes."""
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(clip)
            preds.append(out["ef"].cpu().numpy())
    return np.stack(preds, axis=0)  # (n_passes, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_best.pt")
    n_mc_passes = cfg["uncertainty"]["mc_passes"]
    alpha = cfg["uncertainty"]["conformal_alpha"]  # e.g. 0.1 -> 90% target coverage

    out_dir = Path("results/mc_dropout")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model, enable MC Dropout (dropout active, rest in eval mode) ---
    model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.enable_mc_dropout()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_mae={checkpoint['val_mae']:.3f}")
    print(f"Running MC Dropout with {n_mc_passes} passes per clip...")

    # --- Split: VAL for conformal calibration, TEST for evaluation ---
    # Conformal prediction needs a separate calibration set from the one
    # you report final coverage on -- using VAL for calibration keeps TEST
    # untouched for calibration purposes (though we already looked at TEST
    # metrics in the previous step, which is fine; conformal calibration
    # itself must not reuse TEST).
    def get_loader(split, augment):
        ds = EchoNetDataset(
            cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"],
            split=split, num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
            ef_thresholds=cfg["data"]["ef_thresholds"], augment=augment, seed=cfg["seed"],
        )
        return ds, DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=2)

    cal_ds, cal_loader = get_loader("VAL", AugmentConfig(temporal=False))
    test_ds, test_loader = get_loader("TEST", AugmentConfig(temporal=False))

    def run_mc_dropout_over_loader(loader, n_items, tag):
        all_mean_pred = np.zeros(n_items)
        all_std_pred = np.zeros(n_items)
        all_true = np.zeros(n_items)
        all_filenames = []
        i = 0
        for batch in loader:
            clip = normalize(batch["clip"].to(device), R3D_MEAN, R3D_STD, device)
            bsz = clip.shape[0]
            mc_preds = mc_dropout_predict(model, clip, n_mc_passes)  # (n_passes, bsz)
            all_mean_pred[i:i+bsz] = mc_preds.mean(axis=0)
            all_std_pred[i:i+bsz] = mc_preds.std(axis=0)
            all_true[i:i+bsz] = batch["ef"].numpy()
            all_filenames.extend(batch["filename"])
            i += bsz
            if i % 200 < bsz:
                print(f"  [{tag}] {i}/{n_items} done")
        return all_mean_pred, all_std_pred, all_true, all_filenames

    print("Running on calibration set (VAL)...")
    cal_mean, cal_std, cal_true, _ = run_mc_dropout_over_loader(cal_loader, len(cal_ds), "cal")

    print("Running on test set (TEST)...")
    test_mean, test_std, test_true, test_files = run_mc_dropout_over_loader(test_loader, len(test_ds), "test")

    # === Split conformal prediction ===
    # Nonconformity score = |error| / sigma (normalized residual).
    # q = the (1-alpha) quantile of calibration scores -> interval = pred ± q*sigma
    cal_scores = np.abs(cal_mean - cal_true) / np.clip(cal_std, 1e-3, None)
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(cal_scores, q_level)
    print(f"\nConformal quantile q_hat = {q_hat:.3f} (target coverage {1-alpha:.0%})")

    interval_halfwidth = q_hat * test_std
    lower = test_mean - interval_halfwidth
    upper = test_mean + interval_halfwidth
    covered = (test_true >= lower) & (test_true <= upper)
    empirical_coverage = covered.mean()
    print(f"Empirical test coverage: {empirical_coverage:.1%} (target: {1-alpha:.0%})")
    print(f"Mean interval width: {(2 * interval_halfwidth).mean():.2f} EF points")

    # === Correlation: does higher sigma mean higher error? ===
    test_abs_error = np.abs(test_mean - test_true)
    corr, pval = stats.spearmanr(test_std, test_abs_error)
    print(f"\nSpearman corr(sigma, |error|): {corr:.3f} (p={pval:.2e})")

    # === Selective prediction: MAE as a function of deferral rate ===
    order = np.argsort(-test_std)  # most uncertain first
    deferral_fracs = np.linspace(0, 0.5, 11)  # defer 0% to 50%
    selective_mae = []
    for frac in deferral_fracs:
        n_defer = int(frac * len(test_std))
        keep_idx = order[n_defer:]  # keep the LEAST uncertain ones
        mae_at_frac = test_abs_error[keep_idx].mean() if len(keep_idx) > 0 else np.nan
        selective_mae.append(mae_at_frac)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deferral_fracs * 100, selective_mae, marker="o", color="#1E8E5A")
    ax.set_xlabel("% most-uncertain cases deferred")
    ax.set_ylabel("MAE on remaining cases")
    ax.set_title("Selective prediction: error vs. deferral rate")
    fig.savefig(out_dir / "selective_prediction_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Sigma vs error scatter ===
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(test_std, test_abs_error, s=6, alpha=0.4, color="#1E8E5A")
    ax.set_xlabel("MC Dropout sigma (predictive std)")
    ax.set_ylabel("|Predicted - True EF|")
    ax.set_title(f"Uncertainty vs. error (Spearman r={corr:.3f})")
    fig.savefig(out_dir / "sigma_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Save everything ---
    np.savez(out_dir / "mc_dropout_predictions.npz",
              filenames=test_files, mean_pred=test_mean, std_pred=test_std, true_ef=test_true,
              lower=lower, upper=upper, covered=covered)

    results = {
        "n_mc_passes": n_mc_passes,
        "conformal_alpha": alpha,
        "conformal_q_hat": float(q_hat),
        "empirical_coverage": float(empirical_coverage),
        "target_coverage": float(1 - alpha),
        "mean_interval_width": float((2 * interval_halfwidth).mean()),
        "spearman_sigma_error_corr": float(corr),
        "spearman_pvalue": float(pval),
        "selective_mae_by_deferral": {f"{int(f*100)}%": float(m) for f, m in zip(deferral_fracs, selective_mae)},
    }
    with open(out_dir / "mc_dropout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()