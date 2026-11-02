"""
Read-only diagnostic: why did the k1-vs-k3 label check stop?

The k1 evaluator stores y_true as float32 (via the DataLoader's float32
tensors); the k3 evaluator stores y_true as float64 (straight from the CSV).
This script measures exactly how the saved labels differ - expecting only
float32 rounding - and whether any video's [40, 50] group actually changes.

It reads only the four predictions.npz files and FileList.csv. No writes,
no inference, no TEST data.

Run from the echo-ef-project folder:

    python diagnose_k1_k3_labels.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 1288
THRESHOLDS = (40.0, 50.0)

RUNS = (
    ("R3D k1", "val_r3d18_baseline_fp32_k1"),
    ("R3D k3", "val_r3d18_baseline_fp32_k3"),
    ("Swin k1", "val_swin3d_t_baseline_fp32_k1"),
    ("Swin k3", "val_swin3d_t_baseline_fp32_k3"),
)

PAIRS = (
    ("R3D k1 vs R3D k3", "val_r3d18_baseline_fp32_k1", "val_r3d18_baseline_fp32_k3"),
    ("Swin k1 vs Swin k3", "val_swin3d_t_baseline_fp32_k1", "val_swin3d_t_baseline_fp32_k3"),
)


def stop(message):
    raise SystemExit(f"STOPPED: {message}")


def to_group(values):
    cls = np.zeros(len(values), dtype=int)
    cls += values >= THRESHOLDS[0]
    cls += values >= THRESHOLDS[1]
    return cls


def main():
    parser = argparse.ArgumentParser(description="Read-only k1-vs-k3 label diagnostic")
    parser.add_argument("--results-root", default="results/eval_split")
    parser.add_argument("--file-list", default="data/EchoNet-Dynamic/FileList.csv")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        stop(f"results root not found: {results_root}. Run this from the echo-ef-project folder")

    data = {}
    for label, folder in RUNS:
        path = results_root / folder / "predictions.npz"
        if not path.is_file():
            stop(f"missing {path} - run the missing evaluation first")
        with np.load(path, allow_pickle=False) as z:
            y = np.asarray(z["y_true"])
            names = [str(x) for x in z["file_names"].tolist()]
        if y.shape != (EXPECTED_N,):
            stop(f"{folder}: y_true shape {y.shape}, expected ({EXPECTED_N},)")
        data[folder] = (y, names)

    ref_names = data[RUNS[0][1]][1]
    for label, folder in RUNS[1:]:
        if data[folder][1] != ref_names:
            stop(f"{folder}: video ID order differs from {RUNS[0][1]} - this is a real mismatch, not rounding")

    print("READ-ONLY K1/K3 LABEL DIAGNOSTIC")
    print(f"n={EXPECTED_N} per run; video ID order identical across all four runs: yes")
    print("")

    csv_ok = False
    file_list = Path(args.file_list)
    if file_list.is_file():
        df = pd.read_csv(file_list)
        val = df[df["Split"].str.upper() == "VAL"].reset_index(drop=True)
        if len(val) == EXPECTED_N and val["FileName"].astype(str).tolist() == ref_names:
            csv_ok = True
            csv_ef = val["EF"].to_numpy(dtype=np.float64)
        else:
            print("note: FileList.csv VAL rows do not match the saved ID order; CSV check skipped")
    else:
        print(f"note: {file_list} not found; CSV check skipped")

    for pair_label, f1, f2 in PAIRS:
        y1, y2 = data[f1][0], data[f2][0]
        print(f"--- {pair_label} ---")
        print(f"dtypes: {f1} y_true is {y1.dtype}, {f2} y_true is {y2.dtype}")
        diff = np.abs(y1.astype(np.float64) - y2.astype(np.float64))
        n_diff = int(np.count_nonzero(diff > 0))
        print(f"values differing at all: {n_diff} of {EXPECTED_N}; max abs diff {diff.max():.3e}; "
              f"mean abs diff {diff.mean():.3e}")
        rounding_only = np.array_equal(y1.astype(np.float32), y2.astype(np.float32))
        print(f"differences fully explained by float32 rounding: {rounding_only}")
        g1, g2 = to_group(y1.astype(np.float64)), to_group(y2.astype(np.float64))
        n_flip = int(np.count_nonzero(g1 != g2))
        print(f"group flips at [40, 50]: {n_flip}")
        if n_flip:
            idx = np.nonzero(g1 != g2)[0][:10]
            for i in idx:
                print(f"  {ref_names[i]}: k1 EF {y1[i]!r} group {g1[i]} vs k3 EF {y2[i]!r} group {g2[i]}")
        if csv_ok:
            print(f"matches CSV exactly (float64): k1 {np.array_equal(y1.astype(np.float64), csv_ef)}, "
                  f"k3 {np.array_equal(y2.astype(np.float64), csv_ef)}")
            print(f"matches CSV as float32 rounding: k1 {np.array_equal(y1.astype(np.float32), csv_ef.astype(np.float32))}, "
                  f"k3 {np.array_equal(y2.astype(np.float32), csv_ef.astype(np.float32))}")
        print("")

    print("Reading this: float32-vs-float64 storage rounding with zero group flips means the labels")
    print("agree; the comparison script's exact-equality check was too strict, not the data wrong.")
    print("Any group flip or CSV mismatch above would be a real problem - report it as-is.")
    print("")
    print("Nothing was written.")


if __name__ == "__main__":
    main()
