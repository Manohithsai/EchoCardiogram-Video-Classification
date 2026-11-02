"""
Read-only analysis of the mildly reduced EF cases (true EF in [40, 50)) that the
existing models miss on VAL.

Purpose: describe WHERE the missed mild cases sit, in measurable terms. This
is a measurement step only. It does not establish causes and does not by itself
determine the right fix; any interpretation of the patterns below needs
separate review.

What is measured, all from the saved prediction files (no new computation on
videos, no training, no TEST access, no file writing):

  1. Sub-bin location: how mild recall distributes across 2-point true EF
     sub-bins from 40 to 50. Note: a miss near the 50 boundary can be either
     borderline noise or a fixable error; location alone does not say which.
  2. Miss direction and distance: each missed mild case lands below 40 (called
     reduced) or at/above 50 (called normal); the prediction spread per side.
  3. Cross-model agreement: how many mild videos are missed by exactly 0-4 of
     the four baseline runs. Note: shared misses can reflect common bias across
     similar models as well as genuinely difficult videos; this measures the
     degree of agreement only.
  4. Hard set: which mild videos are missed by ALL four baseline runs (R3D
     k1/k3, Swin k1/k3), and what every model predicts for them.
  5. Data traits of the hard set vs mild cases that are caught (from
     FileList.csv only): frames, FPS, frame size, EDV, ESV. These are
     associations; they can generate hypotheses but do not establish causes.
  6. LDS comparison: which extra mild cases the LDS-trained model catches that
     the R3D baseline misses, and how many extra normal videos it flags to get
     them (fixed EF<50 decision).

Label handling (disclosed): the k1 evaluators stored true EF as float32, the k3
evaluators as float64 (straight from FileList.csv). All runs are compared and
reported in ONE common representation - float32, the least precise storage used -
and the script verifies exact equality of every run's labels and of FileList.csv's
EF column in that representation before computing anything. Any other difference
stops the run.

Ensembles are simple means of saved per-video EF predictions (R3D + Swin, 1/2
each), computed in memory within k1 and within k3. True groups and decision
thresholds stay fixed at [40, 50]. There is no weight search, cutoff search,
model loading, checkpoint loading, inference, training, TEST access, or file
writing.

Run from the echo-ef-project folder:

    python analyze_val_mild_misses.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 1288
THRESHOLDS = (40.0, 50.0)
MILD_BINS = ((40.0, 42.0), (42.0, 44.0), (44.0, 46.0), (46.0, 48.0), (48.0, 50.0))

REQUIRED_RUNS = (
    ("R3D k1", "val_r3d18_baseline_fp32_k1", 1),
    ("R3D k3", "val_r3d18_baseline_fp32_k3", 3),
    ("Swin k1", "val_swin3d_t_baseline_fp32_k1", 1),
    ("Swin k3", "val_swin3d_t_baseline_fp32_k3", 3),
)

# Comparison arms: included when their folder exists, skipped with a note when not.
OPTIONAL_RUNS = (
    ("LDS k1", "val_r3d18_lds_fp32_k1", 1),
    ("CB-DRW k1", "val_r3d18_cb_drw_fp32_k1", 1),
)

REQUIRED_META_FIELDS = (
    "tag",
    "split",
    "n",
    "precision",
    "autocast_active",
    "clips_per_video",
    "checkpoint_sha256_16",
    "predictions_sha256_16",
    "video_ids_sha256_16",
    "normalisation_source",
    "eval_config_sha256_16",
    "file_list_csv_sha256_16",
    "mae",
    "rmse",
    "r2_coefficient_of_determination",
)


def stop(message):
    raise SystemExit(f"STOPPED: {message}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_1d(name, value):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape != (EXPECTED_N,):
        stop(f"{name} must be a 1-D array with shape ({EXPECTED_N},), got {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        stop(f"{name} contains non-finite values")
    return arr


def to_group(values):
    cls = np.zeros(len(values), dtype=int)
    cls += values >= THRESHOLDS[0]
    cls += values >= THRESHOLDS[1]
    return cls


def safe_div(num, den):
    return (num / den) if den else None


def fmt_pct(value):
    return "  n/a" if value is None else f"{100.0 * value:6.1f}"


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = None if sst <= 0 else float(1.0 - np.sum(err ** 2) / sst)
    return mae, rmse, r2


def load_run(results_root, label, folder, expected_clips):
    run_dir = results_root / folder
    if not run_dir.is_dir():
        stop(f"{label} folder not found: {run_dir}")
    metrics_path = run_dir / "metrics.json"
    pred_path = run_dir / "predictions.npz"
    ids_path = run_dir / "video_ids.txt"
    for path in (metrics_path, pred_path, ids_path):
        if not path.is_file():
            stop(f"missing required file: {path}")

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        stop(f"could not read {metrics_path}: {e}")

    missing = [field for field in REQUIRED_META_FIELDS if field not in meta]
    if missing:
        stop(f"{metrics_path} is missing required fields: {missing}")
    if meta["tag"] != folder:
        stop(f"{run_dir} tag mismatch: metrics tag is {meta['tag']!r}")
    if meta["split"] != "VAL":
        stop(f"{run_dir} is split={meta['split']!r}, expected VAL")
    if int(meta["n"]) != EXPECTED_N:
        stop(f"{run_dir} has n={meta['n']}, expected {EXPECTED_N}")
    if meta["precision"] != "fp32" or meta["autocast_active"] is not False:
        stop(f"{run_dir} is not a saved FP32 run")
    if int(meta["clips_per_video"]) != expected_clips:
        stop(f"{run_dir} records clips_per_video={meta['clips_per_video']}, expected {expected_clips}")

    pred_hash = sha256_file(pred_path)[:16]
    ids_hash = sha256_file(ids_path)[:16]
    if pred_hash != meta["predictions_sha256_16"]:
        stop(f"{pred_path} hash does not match metrics.json")
    if ids_hash != meta["video_ids_sha256_16"]:
        stop(f"{ids_path} hash does not match metrics.json")

    try:
        with np.load(pred_path, allow_pickle=False) as z:
            for key in ("y_true", "y_pred", "file_names"):
                if key not in z.files:
                    stop(f"{pred_path} is missing array {key!r}")
            y_true = finite_1d(f"{run_dir}/y_true", z["y_true"])
            y_pred = finite_1d(f"{run_dir}/y_pred", z["y_pred"])
            raw_names = z["file_names"]
            if raw_names.ndim != 1 or raw_names.shape != (EXPECTED_N,):
                stop(f"{run_dir}/file_names has shape {raw_names.shape}, expected ({EXPECTED_N},)")
            names = [str(x) for x in raw_names.tolist()]
    except (OSError, ValueError) as e:
        stop(f"could not read {pred_path}: {e}")

    with open(ids_path, "r", encoding="utf-8") as f:
        file_lines = [line.rstrip("\r\n") for line in f]
    while file_lines and file_lines[-1] == "":
        file_lines.pop()
    if names != file_lines:
        stop(f"{ids_path} does not exactly match file_names in {pred_path}")
    if len(set(names)) != EXPECTED_N:
        stop(f"{run_dir} has duplicate video IDs")

    mae, rmse, r2 = regression_metrics(y_true, y_pred)
    if abs(mae - float(meta["mae"])) > 1e-6:
        stop(f"{run_dir} stored MAE does not match predictions.npz")
    if abs(rmse - float(meta["rmse"])) > 1e-6:
        stop(f"{run_dir} stored RMSE does not match predictions.npz")
    stored_r2 = meta["r2_coefficient_of_determination"]
    if (r2 is None) != (stored_r2 is None) or (r2 is not None and abs(r2 - float(stored_r2)) > 1e-6):
        stop(f"{run_dir} stored true R2 does not match predictions.npz")

    return {
        "label": label,
        "folder": folder,
        "run_dir": run_dir,
        "meta": meta,
        "y_true": y_true,
        "y_pred": y_pred,
        "names": names,
        "pred_hash": pred_hash,
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only analysis of missed mildly reduced EF cases on VAL")
    parser.add_argument("--results-root", default="results/eval_split")
    parser.add_argument("--file-list", default="data/EchoNet-Dynamic/FileList.csv")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        stop(f"results root not found: {results_root}. Run this from the echo-ef-project folder")

    runs = [load_run(results_root, label, folder, clips) for label, folder, clips in REQUIRED_RUNS]
    for label, folder, clips in OPTIONAL_RUNS:
        if (results_root / folder).is_dir():
            runs.append(load_run(results_root, label, folder, clips))
        else:
            print(f"NOTE: optional run folder not found, skipping this arm: {results_root / folder}")
    run_by_label = {run["label"]: run for run in runs}

    reference = runs[0]
    for run in runs[1:]:
        if run["names"] != reference["names"]:
            stop(f"{run['label']} video IDs do not match the baseline order")
        for field in ("eval_config_sha256_16", "file_list_csv_sha256_16"):
            if run["meta"][field] != reference["meta"][field]:
                stop(f"{run['label']} {field} does not match the baseline run")

    # One explicit common label representation: float32 (the least precise storage used by
    # the evaluators; the k1 runs stored float32, the k3 runs stored float64 from the CSV).
    labels_common = reference["y_true"].astype(np.float32)
    for run in runs[1:]:
        run_common = run["y_true"].astype(np.float64).astype(np.float32)
        if not np.array_equal(run_common, labels_common):
            stop(f"{run['label']} EF labels differ from {reference['label']} even after conversion "
                 f"to the common float32 representation - this is a real mismatch, not storage rounding")

    # Positive source alignment: FileList.csv EF must equal the common labels exactly (as float32).
    csv_path = Path(args.file_list)
    if not csv_path.is_file():
        stop(f"FileList.csv not found: {csv_path}. Source-label alignment cannot be verified; not proceeding")
    csv_df = pd.read_csv(csv_path)
    csv_val = csv_df[csv_df["Split"].str.upper() == "VAL"].reset_index(drop=True)
    if len(csv_val) != EXPECTED_N or csv_val["FileName"].astype(str).tolist() != reference["names"]:
        stop("FileList.csv VAL rows do not match the saved runs' videos/order; not proceeding")
    csv_common = csv_val["EF"].to_numpy(dtype=np.float64).astype(np.float32)
    if not np.array_equal(csv_common, labels_common):
        stop("saved labels do not match FileList.csv EF even as float32 - this is a real mismatch, not storage rounding")
    y_true = labels_common.astype(np.float64)

    # In-memory fixed 1/2 ensembles, within k1 and within k3.
    ens = {
        "Ens k1": 0.5 * (run_by_label["R3D k1"]["y_pred"] + run_by_label["Swin k1"]["y_pred"]),
        "Ens k3": 0.5 * (run_by_label["R3D k3"]["y_pred"] + run_by_label["Swin k3"]["y_pred"]),
    }
    for label, pred in ens.items():
        if not np.isfinite(pred).all():
            stop(f"{label} prediction contains non-finite values")

    pred_series = [(run["label"], run["y_pred"]) for run in runs] + list(ens.items())
    labels_order = [label for label, _ in pred_series]
    yt_grp = to_group(y_true)
    mild_mask = yt_grp == 1
    n_mild = int(np.count_nonzero(mild_mask))
    yp_grp = {label: to_group(pred) for label, pred in pred_series}
    mild_hit = {label: (yp_grp[label] == 1) & mild_mask for label in labels_order}

    print("READ-ONLY MISSED-MILD ANALYSIS (VAL, n=1288, FP32, fixed groups [40,50])")
    print("No checkpoint, model, training data, or TEST file is read. No files are written.")
    print("Labels: all runs and FileList.csv agree exactly in the common float32 representation")
    print("")
    print("Artifact provenance and checks:")
    for run in runs:
        meta = run["meta"]
        print(
            f"  {run['label']}: folder={run['run_dir']} | checkpoint_sha={meta['checkpoint_sha256_16']} | "
            f"predictions_sha={run['pred_hash']} | clips_per_video={meta['clips_per_video']} | "
            f"normalisation={meta['normalisation_source']}"
        )
    print("  Passed: n=1288, split=VAL, FP32, expected clip counts, unique IDs, identical ID order, saved hashes match")
    print("")
    print(f"Mildly reduced true group (EF in [40,50)): n={n_mild}")
    print("")

    # ---- Section 1: where the mild misses sit (true EF sub-bins) ----
    print("1) Mild recall (%) by true EF sub-bin - where the misses are")
    header = f"{'true EF bin':<12} {'n':>4} | " + " ".join(f"{label:>8}" for label in labels_order)
    print(header)
    for lo, hi in MILD_BINS:
        bin_mask = mild_mask & (y_true >= lo) & (y_true < hi)
        n_bin = int(np.count_nonzero(bin_mask))
        cells = []
        for label in labels_order:
            caught = int(np.count_nonzero(mild_hit[label] & bin_mask))
            cells.append(fmt_pct(safe_div(caught, n_bin)))
        print(f"[{lo:4.0f},{hi:4.0f})    {n_bin:>4} | " + " ".join(cells))
    all_cells = [fmt_pct(safe_div(int(np.count_nonzero(mild_hit[label])), n_mild)) for label in labels_order]
    print(f"{'all mild':<12} {n_mild:>4} | " + " ".join(all_cells))
    print("")

    # ---- Section 2: miss direction and distance - missed mild cases split by side ----
    pred_by_label = dict(pred_series)
    print("2) Miss direction and distance - a missed mild case lands on one of two sides:")
    print("   below 40 (called reduced) or at/above 50 (called normal). Stats per side:")
    print(f"{'run':<10} | {'miss':>4} | {'<40_n':>5} {'med':>6} {'min':>6} | {'>=50_n':>6} {'med':>6} {'max':>6} {'<52':>5} {'>=55':>5}")
    for label in labels_order:
        miss_mask = mild_mask & (yp_grp[label] != 1)
        low = miss_mask & (yp_grp[label] == 0)
        high = miss_mask & (yp_grp[label] == 2)
        pred = pred_by_label[label]
        n_miss = int(np.count_nonzero(miss_mask))
        n_low, n_high = int(np.count_nonzero(low)), int(np.count_nonzero(high))
        low_med = f"{float(np.median(pred[low])):6.2f}" if n_low else "   n/a"
        low_min = f"{float(np.min(pred[low])):6.2f}" if n_low else "   n/a"
        high_med = f"{float(np.median(pred[high])):6.2f}" if n_high else "   n/a"
        high_max = f"{float(np.max(pred[high])):6.2f}" if n_high else "   n/a"
        pct_lt52 = f"{100.0 * np.count_nonzero(pred[high] < 52.0) / n_high:>4.0f}%" if n_high else "  n/a"
        pct_ge55 = f"{100.0 * np.count_nonzero(pred[high] >= 55.0) / n_high:>4.0f}%" if n_high else "  n/a"
        print(f"{label:<10} | {n_miss:>4} | {n_low:>5} {low_med} {low_min} | "
              f"{n_high:>6} {high_med} {high_max} {pct_lt52} {pct_ge55}")
    print("")

    # ---- Section 3: cross-model agreement on mild misses (4 baseline runs) ----
    base_labels = ["R3D k1", "R3D k3", "Swin k1", "Swin k3"]
    base_miss = np.zeros((len(base_labels), EXPECTED_N), dtype=bool)
    for i, label in enumerate(base_labels):
        base_miss[i] = mild_mask & (yp_grp[label] != 1)
    miss_count_full = base_miss.sum(axis=0)
    miss_count = miss_count_full[mild_mask]
    print("3) Cross-model agreement on mild misses (baseline runs: R3D k1/k3, Swin k1/k3)")
    for k in range(5):
        n_k = int(np.count_nonzero(miss_count == k))
        note = ""
        if k == 0:
            note = "  <- caught by all 4"
        elif k == 4:
            note = "  <- missed by all 4 (hard set)"
        print(f"   missed by exactly {k} of 4: {n_k:>4}{note}")
    union_miss = np.count_nonzero(miss_count >= 1)
    inter_miss = np.count_nonzero(miss_count == 4)
    print(f"   union of misses: {int(union_miss)} | intersection (hard set): {int(inter_miss)}")
    print("")

    # ---- Section 4: the hard set, per video ----
    hard_mask = mild_mask & (miss_count_full == 4)
    hard_idx = np.nonzero(hard_mask)[0]
    hard_idx = hard_idx[np.argsort(y_true[hard_idx])]
    print(f"4) Hard set: mild videos missed by ALL 4 baseline runs (n={len(hard_idx)}), sorted by true EF")
    print(f"   {'video':<22} {'trueEF':>6} " + " ".join(f"{label:>7}" for label in labels_order))
    for idx in hard_idx:
        cells = " ".join(f"{pred_series[j][1][idx]:>7.2f}" for j in range(len(labels_order)))
        print(f"   {reference['names'][idx]:<22} {y_true[idx]:>6.2f} {cells}")
    print("")

    # ---- Section 5: data traits of the hard set vs caught mild cases (FileList.csv only) ----
    trait_cols = [c for c in ("NumberOfFrames", "FPS", "FrameHeight", "FrameWidth", "EDV", "ESV")
                  if c in csv_val.columns]
    caught_mask = mild_mask & (miss_count_full <= 2)
    if trait_cols:
        print("5) Video traits (FileList.csv): hard set (missed by all 4) vs mild cases caught by at least 3 of 4")
        print(f"   {'group':<28} {'n':>4} " + " ".join(f"{c + '_mean':>14}" for c in trait_cols))
        for group_label, mask in (("missed by all 4", hard_mask), ("caught by >=3 of 4", caught_mask)):
            sub = csv_val[mask]
            cells = " ".join(f"{float(pd.to_numeric(sub[c], errors='coerce').mean()):>14.2f}" for c in trait_cols)
            print(f"   {group_label:<28} {int(np.count_nonzero(mask)):>4} {cells}")
        print("")
    else:
        print("5) Video traits: none of NumberOfFrames/FPS/FrameHeight/FrameWidth/EDV/ESV present in FileList.csv; skipped")
        print("")

    # ---- Section 6: LDS trade-off in detail (skipped if the arm was not loaded) ----
    if "LDS k1" in run_by_label:
        lds_pred = run_by_label["LDS k1"]["y_pred"]
        r3d1_pred = run_by_label["R3D k1"]["y_pred"]
        lds_pos = lds_pred < THRESHOLDS[1]
        r3d1_pos = r3d1_pred < THRESHOLDS[1]
        truth_pos50 = y_true < THRESHOLDS[1]
        extra_mild = mild_mask & lds_pos & ~r3d1_pos
        extra_normal = (yt_grp == 2) & lds_pos & ~r3d1_pos
        lost_mild = mild_mask & ~lds_pos & r3d1_pos
        print("6) LDS k1 vs R3D k1 at the fixed EF<50 decision")
        print(f"   R3D k1:  TP={int(np.count_nonzero(truth_pos50 & r3d1_pos))} "
              f"FP={int(np.count_nonzero(~truth_pos50 & r3d1_pos))}")
        print(f"   LDS k1:  TP={int(np.count_nonzero(truth_pos50 & lds_pos))} "
              f"FP={int(np.count_nonzero(~truth_pos50 & lds_pos))}")
        print(f"   mild cases caught by LDS but missed by R3D k1: {int(np.count_nonzero(extra_mild))}")
        for lo, hi in MILD_BINS:
            n_bin = int(np.count_nonzero(extra_mild & (y_true >= lo) & (y_true < hi)))
            print(f"     of these, true EF [{lo:4.0f},{hi:4.0f}): {n_bin}")
        print(f"   mild cases caught by R3D k1 but missed by LDS: {int(np.count_nonzero(lost_mild))}")
        print(f"   extra normal videos flagged by LDS (true normal, LDS<50, R3D k1>=50): {int(np.count_nonzero(extra_normal))}")
        print("")
    else:
        print("6) LDS comparison skipped: LDS run folder was not available")
        print("")

    print("Descriptive only: no weight search, no cutoff search, no significance test. "
          "Ensembles are computed in memory and are not saved.")
    print("Locations, agreement counts and trait differences are observations, not established causes; "
          "deciding what they mean and what to do next is a separate step.")


if __name__ == "__main__":
    main()
