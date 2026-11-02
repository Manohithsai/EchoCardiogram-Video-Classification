"""
Read-only feasibility check for multi-clip evaluation.

Current single-clip evaluation takes one centred window of
span = num_frames * frame_stride frames (32 * 2 = 64) from each cached
video. Multi-clip evaluation would average predictions over several
windows per video. The span used below is the current evaluation window
(num_frames x frame_stride, a conservative fixed window). Fitting more
windows only means the frames exist - it does not prove the extra windows
show different heart motion, and it does not by itself mean multi-clip
will improve accuracy. That is what a later evaluation run would test.

This script only reads file headers (memory-mapped) and FileList.csv.
It runs no inference, loads no checkpoints, touches no TEST data and
writes nothing.

Run from the echo-ef-project folder (defaults match configs/base.yaml):

    python check_multiclip_cache.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SPLIT = "VAL"
EXPECTED_N = 1288


def stop(message):
    raise SystemExit(f"STOPPED: {message}")


def main():
    parser = argparse.ArgumentParser(description="Read-only multi-clip cache feasibility check")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--file-list", default="data/EchoNet-Dynamic/FileList.csv")
    parser.add_argument("--num-frames", type=int, default=32, help="must match configs/base.yaml data.num_frames")
    parser.add_argument("--frame-stride", type=int, default=2, help="must match configs/base.yaml data.frame_stride")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    file_list = Path(args.file_list)
    if not cache_dir.is_dir():
        stop(f"cache dir not found: {cache_dir}. Run this from the echo-ef-project folder")
    if not file_list.is_file():
        stop(f"FileList.csv not found: {file_list}. Run this from the echo-ef-project folder")

    df = pd.read_csv(file_list)
    for col in ("FileName", "Split"):
        if col not in df.columns:
            stop(f"FileList.csv is missing required column {col!r}; found {list(df.columns)}")
    val = df[df["Split"].str.upper() == EXPECTED_SPLIT].reset_index(drop=True)
    if len(val) != EXPECTED_N:
        stop(f"VAL rows = {len(val)}, expected {EXPECTED_N}. Do not continue; the split does not match our saved runs")

    span = args.num_frames * args.frame_stride
    lengths = []
    shapes = set()
    dtypes = set()
    missing = []
    for fn in val["FileName"]:
        path = cache_dir / f"{Path(fn).stem}.npy"
        if not path.is_file():
            missing.append(fn)
            continue
        try:
            arr = np.load(path, mmap_mode="r")  # header only; no pixel data read into memory
        except (OSError, ValueError) as e:
            stop(f"could not read {path}: {e}")
        if arr.ndim != 3:
            stop(f"{path} has shape {arr.shape}, expected (T, H, W)")
        shapes.add((arr.shape[1], arr.shape[2]))
        dtypes.add(str(arr.dtype))
        lengths.append(int(arr.shape[0]))

    if missing:
        stop(f"{len(missing)} cached .npy files missing for VAL (e.g. {missing[:3]})")

    lengths = np.array(lengths)
    n = len(lengths)
    print("READ-ONLY MULTI-CLIP FEASIBILITY CHECK")
    print(f"cache_dir={cache_dir} | split={EXPECTED_SPLIT} | videos={n}")
    print(f"clip window = num_frames {args.num_frames} x frame_stride {args.frame_stride} = span {span} frames")
    print("(confirm num_frames/frame_stride match configs/base.yaml before trusting this)")
    print("")
    if shapes != {(112, 112)}:
        print(f"note: frame sizes found = {sorted(shapes)}")
    if dtypes != {"uint8"}:
        print(f"note: dtypes found = {sorted(dtypes)}")
    print(f"frames per video: min={lengths.min()} p25={int(np.percentile(lengths, 25))} "
          f"median={int(np.median(lengths))} p75={int(np.percentile(lengths, 75))} max={lengths.max()}")
    print("")
    n_short = int(np.count_nonzero(lengths < span))
    print(f"videos shorter than one span ({span} frames): {n_short} of {n} "
          f"({100.0 * n_short / n:.1f}%) - even one clip is loop-padded (repeats identical frames) for these")
    print("")
    print("max NON-OVERLAPPING (zero shared frames) windows that fit per video (floor(total_frames / span)):")
    for k in (1, 2, 3, 5, 8):
        n_fit = int(np.count_nonzero(lengths >= k * span))
        print(f"  {k} window(s): {n_fit} of {n} videos ({100.0 * n_fit / n:.1f}%)")
    print("")
    windows = np.floor(lengths / span).astype(int)
    print(f"non-overlapping windows available per video: median={int(np.median(windows))} "
          f"min={windows.min()} max={windows.max()}")
    print("")
    print("How to read this:")
    print("- Videos below a given line can still be sampled with OVERLAPPING windows; overlapping windows")
    print("  share some frames but not all. Only loop-padding (videos shorter than one span) reuses identical frames.")
    print("- Fitting more windows only means the frames exist. It does not prove the extra windows show")
    print("  different heart motion, and it does not by itself mean multi-clip improves accuracy.")
    print("  That question is what a later multi-clip evaluation run would test.")
    print("")
    print("Nothing was written. This does not run any model.")


if __name__ == "__main__":
    main()
