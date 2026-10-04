"""
Quantitative Grad-CAM validation: does the model's attention actually
peak near the expert-annotated ED/ES frames from VolumeTracings.csv?
Tests this against a random-baseline hit rate, and saves a few example
heatmap visualizations.

Usage:
    python -m src.xai.validate_gradcam --config configs/base.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask
from src.xai.grad_cam_3d import GradCAM3D

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]
TOLERANCE_FRAMES = 15  # ~0.3s at 50fps -- a cardiac-cycle-scale window around each annotated frame
N_VIZ_EXAMPLES = 3


def compute_sector_mask(raw_clip: np.ndarray, cam_shape: tuple) -> torch.Tensor:
    """Derives the ultrasound sector (the fan-shaped region of real scan
    data) from the raw video's pixel content -- anything consistently
    near-black across the clip is background/corner, not anatomy. This
    adapts automatically to the coarse resolution of layer4's feature
    map, unlike a fixed border-fraction crop which rounded to a 0-pixel
    margin at this resolution and had no effect.
    Returns a (1, H', W') mask at the CAM's spatial resolution, broadcastable
    across the T' dimension."""
    T_prime, H_prime, W_prime = cam_shape
    mean_frame = raw_clip.mean(axis=0)  # (H, W) -- average over time; sector position is static
    sector = (mean_frame > 10).astype(np.float32)  # non-black -> inside the sector
    sector_t = torch.from_numpy(sector).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sector_small = torch.nn.functional.interpolate(sector_t, size=(H_prime, W_prime), mode="area")
    sector_small = (sector_small > 0.3).float()  # re-binarize after downsampling/blurring
    return sector_small.squeeze(0)  # (1, H', W')


def apply_sector_mask(cam: torch.Tensor, raw_clip: np.ndarray) -> torch.Tensor:
    """Zeroes out CAM activity outside the real ultrasound sector."""
    sector_mask = compute_sector_mask(raw_clip, cam.shape)  # (1, H', W')
    return cam * sector_mask  # broadcasts over T'


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def get_keyframes(tracings_df: pd.DataFrame, filename: str) -> list:
    """The two expert-annotated frame numbers (ED and ES) for this video.
    VolumeTracings.csv stores FileName with a .avi suffix; FileList.csv
    (and our dataset's item['filename']) does not -- normalize both to
    the bare stem before comparing."""
    stem = Path(filename).stem
    rows = tracings_df[tracings_df["FileName"].str.replace(".avi", "", regex=False) == stem]
    return sorted(rows["Frame"].unique().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to checkpoints/r3d18_mcdropout_best.pt")
    parser.add_argument("--n_samples", type=int, default=300,
                         help="Number of test clips to evaluate (full test set is slower; Grad-CAM needs a backward pass per clip)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["checkpoint_dir"]) / "r3d18_mcdropout_best.pt")

    out_dir = Path("results/gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "examples"
    viz_dir.mkdir(exist_ok=True)

    model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # eval mode: dropout off, so Grad-CAM reflects deterministic attention
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_mae={ckpt['val_mae']:.3f}")

    cam_extractor = GradCAM3D(model, model.backbone.layer4)

    tracings_df = pd.read_csv(cfg["data"]["volume_tracings_csv"])

    test_ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="TEST",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )

    n_samples = min(args.n_samples, len(test_ds))
    rng = np.random.default_rng(cfg["seed"])
    sample_indices = rng.choice(len(test_ds), size=n_samples, replace=False)

    hits, min_distances, errors = [], [], []
    viz_saved = 0

    for count, idx in enumerate(sample_indices):
        item = test_ds[idx]
        filename = item["filename"]
        stem = Path(filename).stem

        keyframes = get_keyframes(tracings_df, filename)
        if len(keyframes) < 2:
            continue  # no tracing annotation for this video -- skip

        # Re-derive the exact original-video frame indices this clip used
        # (center-sampled, deterministic for non-train splits).
        total_frames = np.load(Path(cfg["data"]["cache_dir"]) / f"{stem}.npy").shape[0]
        clip_frame_indices = test_ds._sample_clip_indices(total_frames)

        clip = item["clip"].unsqueeze(0).to(device)
        clip = normalize(clip, R3D_MEAN, R3D_STD, device)
        clip.requires_grad_(False)  # we only need gradients w.r.t. activations, not the input

        cam_raw = cam_extractor.generate(clip)  # (T', H', W') -- unmasked, used for visualization
        raw_clip_full = np.load(Path(cfg["data"]["cache_dir"]) / f"{stem}.npy")  # needed for the sector mask
        cam = apply_sector_mask(cam_raw, raw_clip_full)
        temporal_importance = cam.sum(dim=(1, 2)).numpy()
        peak_bin = int(np.argmax(temporal_importance))

        T_prime = cam.shape[0]
        bin_width = len(clip_frame_indices) / T_prime
        bin_center_in_clip = int((peak_bin + 0.5) * bin_width)
        bin_center_in_clip = min(bin_center_in_clip, len(clip_frame_indices) - 1)
        attended_original_frame = int(clip_frame_indices[bin_center_in_clip])

        dists = [abs(attended_original_frame - kf) for kf in keyframes]
        min_dist = min(dists)
        hit = min_dist <= TOLERANCE_FRAMES

        hits.append(hit)
        min_distances.append(min_dist)
        errors.append(abs(item["ef"].item() - ckpt["val_mae"]))  # placeholder not used for stats; real pred below

        if viz_saved < N_VIZ_EXAMPLES:
            fig, axes = plt.subplots(1, T_prime, figsize=(3 * T_prime, 3.2))
            raw_clip = raw_clip_full  # already loaded above, reused here
            for t in range(T_prime):
                frame_in_clip = min(int((t + 0.5) * bin_width), len(clip_frame_indices) - 1)
                orig_frame_idx = clip_frame_indices[frame_in_clip]
                base_img = raw_clip[orig_frame_idx]
                heat = cam_raw[t].numpy()
                heat_resized = np.array(
                    torch.nn.functional.interpolate(
                        torch.tensor(heat).unsqueeze(0).unsqueeze(0),
                        size=base_img.shape, mode="bilinear", align_corners=False
                    ).squeeze()
                )
                ax = axes[t] if T_prime > 1 else axes
                ax.imshow(base_img, cmap="gray")
                ax.imshow(heat_resized, cmap="jet", alpha=0.45)
                is_near_key = min(abs(orig_frame_idx - kf) for kf in keyframes) <= TOLERANCE_FRAMES
                is_peak_attention = (t == peak_bin)
                label = f"frame {orig_frame_idx}"
                if is_peak_attention:
                    label += " [peak attn]"
                if is_near_key:
                    label += " [near ED/ES]"
                title_color = "#1E8E5A" if (is_peak_attention and is_near_key) else (
                    "#C25A2E" if is_peak_attention else ("#1E8E5A" if is_near_key else "#14213D"))
                ax.set_title(label, color=title_color, fontsize=8)
                ax.axis("off")
            fig.suptitle(f"{stem} | keyframes={keyframes} | hit={hit}")
            fig.savefig(viz_dir / f"{stem}_gradcam.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            viz_saved += 1

        if (count + 1) % 50 == 0:
            print(f"  {count+1}/{n_samples} processed...")

    hits = np.array(hits)
    hit_rate = hits.mean()

    # Random baseline: Monte Carlo over the same clips' keyframes, picking
    # a uniformly random "attended frame" within each clip's valid range.
    rng2 = np.random.default_rng(cfg["seed"] + 1)
    random_hits = []
    for idx in sample_indices:
        item = test_ds[idx]
        filename = item["filename"]
        stem = Path(filename).stem
        keyframes = get_keyframes(tracings_df, filename)
        if len(keyframes) < 2:
            continue
        total_frames = np.load(Path(cfg["data"]["cache_dir"]) / f"{stem}.npy").shape[0]
        random_frame = rng2.integers(0, total_frames)
        dist = min(abs(random_frame - kf) for kf in keyframes)
        random_hits.append(dist <= TOLERANCE_FRAMES)
    random_hit_rate = np.mean(random_hits)

    print(f"\n=== Grad-CAM temporal alignment ===")
    print(f"Samples evaluated: {len(hits)}")
    print(f"Model hit rate (within {TOLERANCE_FRAMES} frames of ED/ES): {hit_rate:.1%}")
    print(f"Random baseline hit rate: {random_hit_rate:.1%}")
    print(f"Lift over random: {hit_rate - random_hit_rate:+.1%}")
    print(f"Mean distance to nearest keyframe: {np.mean(min_distances):.1f} frames")

    results = {
        "n_samples": int(len(hits)),
        "tolerance_frames": TOLERANCE_FRAMES,
        "model_hit_rate": float(hit_rate),
        "random_baseline_hit_rate": float(random_hit_rate),
        "lift_over_random": float(hit_rate - random_hit_rate),
        "mean_distance_to_keyframe": float(np.mean(min_distances)),
    }
    with open(out_dir / "gradcam_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/, example heatmaps in {viz_dir}/")


if __name__ == "__main__":
    main()