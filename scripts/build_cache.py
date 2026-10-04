"""
Decode every EchoNet-Dynamic AVI once into a resized uint8 .npy array.

Why: cv2.VideoCapture decoding on the fly is the actual training bottleneck
on a laptop GPU, not the model forward pass. Doing this once up front makes
every later epoch fast and makes the Dataset class trivial (just np.load).

Usage:
    python scripts/build_cache.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_video_as_array(video_path: Path, out_size: int) -> np.ndarray:
    """Reads an AVI and returns (T, H, W) uint8 grayscale frames resized to out_size."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # EchoNet AVIs are already near-grayscale; convert defensively.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Zero frames decoded from {video_path}")

    return np.stack(frames, axis=0).astype(np.uint8)  # (T, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of videos, for a quick smoke test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    raw_dir = Path(data_cfg["raw_video_dir"])
    file_list_path = Path(data_cfg["file_list_csv"])
    cache_dir = Path(data_cfg["cache_dir"])
    out_size = data_cfg["frame_size"]

    if not file_list_path.exists():
        print(f"FileList.csv not found at {file_list_path}.", file=sys.stderr)
        print(
            "Register at Stanford AIMI and download EchoNet-Dynamic first. "
            "See README.md for the access process.",
            file=sys.stderr,
        )
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(file_list_path)

    if args.limit:
        df = df.head(args.limit)

    n_ok, n_fail = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching videos"):
        filename = row["FileName"]
        stem = Path(filename).stem
        out_path = cache_dir / f"{stem}.npy"

        if out_path.exists():
            n_ok += 1
            continue  # resumable: re-running skips what's already cached

        video_path = raw_dir / filename
        if not video_path.suffix:
            video_path = raw_dir / f"{filename}.avi"

        try:
            arr = read_video_as_array(video_path, out_size)
            np.save(out_path, arr)
            n_ok += 1
        except Exception as e:
            print(f"[WARN] Failed on {filename}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Done. Cached OK: {n_ok}, Failed: {n_fail}")
    if n_fail > 0:
        print(
            "Investigate failures before training -- silently dropping videos "
            "can bias your split.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
