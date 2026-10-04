"""
Exploratory data analysis for EchoNet-Dynamic.
Run once after the cache is built. Produces plots + a printed summary
so you actually know your data before training anything.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ef_to_class(ef, thresholds):
    cls = 0
    for t in thresholds:
        if ef >= t:
            cls += 1
        else:
            break
    return cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    thresholds = data_cfg["ef_thresholds"]

    out_dir = Path("results/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_cfg["file_list_csv"])
    print(f"Total videos: {len(df)}")
    print("\nSplit counts:")
    print(df["Split"].value_counts())

    # --- EF class distribution ---
    class_names = ["Reduced (<40)", "Mildly reduced (40-49)", "Normal (>=50)"]
    df["ef_class"] = df["EF"].apply(lambda ef: ef_to_class(ef, thresholds))
    print("\nEF class counts:")
    for i, name in enumerate(class_names):
        count = (df["ef_class"] == i).sum()
        pct = 100 * count / len(df)
        print(f"  {name}: {count} ({pct:.1f}%)")

    # --- EF histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["EF"], bins=40, color="#1E8E5A", edgecolor="white")
    ax.axvline(40, color="#C25A2E", linestyle="--", label="EF=40")
    ax.axvline(50, color="#C25A2E", linestyle="--", label="EF=50")
    ax.set_xlabel("Ejection Fraction (%)")
    ax.set_ylabel("Count")
    ax.set_title("EF distribution across EchoNet-Dynamic")
    ax.legend()
    fig.savefig(out_dir / "ef_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Video length / FPS distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["NumberOfFrames"], bins=40, color="#14213D", edgecolor="white")
    axes[0].set_title("Frames per video")
    axes[0].set_xlabel("NumberOfFrames")

    axes[1].hist(df["FPS"], bins=40, color="#14213D", edgecolor="white")
    axes[1].set_title("FPS distribution")
    axes[1].set_xlabel("FPS")
    fig.savefig(out_dir / "length_fps_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- EF vs EDV/ESV sanity check ---
    implied_ef = 100 * (df["EDV"] - df["ESV"]) / df["EDV"]
    diff = (df["EF"] - implied_ef).abs()
    print(f"\nEF vs implied-EF-from-volumes: mean abs diff = {diff.mean():.3f}, max = {diff.max():.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["EF"], implied_ef, s=4, alpha=0.3, color="#1E8E5A")
    ax.plot([0, 100], [0, 100], color="#C25A2E", linestyle="--")
    ax.set_xlabel("Reported EF")
    ax.set_ylabel("Implied EF from (EDV-ESV)/EDV")
    ax.set_title("Label consistency check")
    fig.savefig(out_dir / "ef_consistency_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()