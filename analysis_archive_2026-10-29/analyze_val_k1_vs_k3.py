"""
Read-only paired comparison of single-clip (k1) and 3-clip (k3) VAL evaluations.

Question: does averaging 3 evenly spaced clips per video change the two ensemble
members, and what does the fixed equal-weight R3D+Swin average look like at k1
versus k3? Everything is descriptive on the same VAL videos and labels; this is
not a significance test, a weight search, or a cutoff search.

Label handling (disclosed): the k1 evaluator stored true EF as float32, the k3
evaluator as float64 (straight from FileList.csv). All runs are compared and
reported in ONE common representation - float32, the least precise storage used -
and the script verifies exact equality of every run's labels and of FileList.csv's
EF column in that representation before computing anything. Any other difference
stops the run.

Ensemble predictions are simple means of saved per-video EF predictions, 1/2
weight each, computed separately within k1 and within k3. True groups and
decision thresholds stay fixed at [40, 50]. There is no model loading,
checkpoint loading, inference, training, TEST access, or file writing.

Run from the echo-ef-project folder:

    python analyze_val_k1_vs_k3.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 1288
THRESHOLDS = (40.0, 50.0)

RUN_FOLDERS = (
    ("R3D baseline k1", "val_r3d18_baseline_fp32_k1", 1),
    ("R3D baseline k3", "val_r3d18_baseline_fp32_k3", 3),
    ("Swin baseline k1", "val_swin3d_t_baseline_fp32_k1", 1),
    ("Swin baseline k3", "val_swin3d_t_baseline_fp32_k3", 3),
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
    return "n/a" if value is None else f"{100.0 * value:5.1f}"


def fmt_bias(value):
    return "n/a" if value is None else f"{value:+5.2f}"


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


def binary_counts(y_true, y_pred, cutoff):
    truth_pos = y_true < cutoff
    pred_pos = y_pred < cutoff
    tp = int(np.count_nonzero(truth_pos & pred_pos))
    fp = int(np.count_nonzero(~truth_pos & pred_pos))
    n_pos = int(np.count_nonzero(truth_pos))
    n_neg = int(np.count_nonzero(~truth_pos))
    return {
        "tp": tp,
        "fp": fp,
        "sensitivity": safe_div(tp, n_pos),
        "specificity": safe_div(n_neg - fp, n_neg),
    }


def all_metrics(y_true, y_pred):
    mae, rmse, r2 = regression_metrics(y_true, y_pred)
    yt_cls = to_group(y_true)
    yp_cls = to_group(y_pred)
    groups = []
    f1_values = []
    for cls in range(3):
        true_mask = yt_cls == cls
        pred_mask = yp_cls == cls
        support = int(np.count_nonzero(true_mask))
        predicted = int(np.count_nonzero(pred_mask))
        tp = int(np.count_nonzero(true_mask & pred_mask))
        recall = safe_div(tp, support)
        precision = safe_div(tp, predicted)
        bias = float(np.mean((y_pred - y_true)[true_mask])) if support else None
        f1 = safe_div(2 * tp, support + predicted)
        if f1 is not None:
            f1_values.append(f1)
        groups.append({"support": support, "recall": recall, "precision": precision, "bias": bias})
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy": float(np.count_nonzero(yt_cls == yp_cls) / len(y_true)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "groups": groups,
        "ef_lt_40": binary_counts(y_true, y_pred, THRESHOLDS[0]),
        "ef_lt_50": binary_counts(y_true, y_pred, THRESHOLDS[1]),
    }


def print_provenance(runs):
    print("READ-ONLY PAIRED VAL COMPARISON: 1 centred clip (k1) vs 3 evenly spaced clips, averaged (k3)")
    print("Ensembles = mean(R3D baseline, Swin baseline) with fixed 1/2 weights, computed separately within k1 and k3.")
    print("True groups and decision thresholds stay fixed at [40, 50]. No weight or cutoff search; descriptive only.")
    print("No checkpoint, model, training data, or TEST file is read. No files are written.")
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
    print("  Labels: all runs and FileList.csv agree exactly in the common float32 representation")
    print("  (k1 stored float32, k3 stored float64; metrics below use that one shared label set)")
    print("")


def print_result_row(label, metrics):
    g = metrics["groups"]
    d40 = metrics["ef_lt_40"]
    d50 = metrics["ef_lt_50"]
    print(
        f"{label:<24} | {metrics['mae']:5.3f} {metrics['rmse']:5.3f} {metrics['r2']:5.3f} | "
        f"{metrics['accuracy']:.3f} {metrics['macro_f1']:.3f} | "
        f"{fmt_pct(g[0]['recall'])}/{fmt_pct(g[0]['precision'])}/{fmt_bias(g[0]['bias'])} | "
        f"{fmt_pct(g[1]['recall'])}/{fmt_pct(g[1]['precision'])}/{fmt_bias(g[1]['bias'])} | "
        f"{fmt_pct(g[2]['recall'])}/{fmt_pct(g[2]['precision'])}/{fmt_bias(g[2]['bias'])} | "
        f"{d40['tp']:3d}/{d40['fp']:3d} {fmt_pct(d40['sensitivity'])}/{fmt_pct(d40['specificity'])} | "
        f"{d50['tp']:3d}/{d50['fp']:3d} {fmt_pct(d50['sensitivity'])}/{fmt_pct(d50['specificity'])}"
    )


def main():
    parser = argparse.ArgumentParser(description="Read-only equal-weight ensemble comparison on saved VAL predictions")
    parser.add_argument("--results-root", default="results/eval_split")
    parser.add_argument("--file-list", default="data/EchoNet-Dynamic/FileList.csv")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        stop(f"results root not found: {results_root}. Run this from the echo-ef-project folder")

    runs = [load_run(results_root, label, folder, clips) for label, folder, clips in RUN_FOLDERS]
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

    ens_k1 = 0.5 * (runs[0]["y_pred"] + runs[2]["y_pred"])
    ens_k3 = 0.5 * (runs[1]["y_pred"] + runs[3]["y_pred"])
    if not np.isfinite(ens_k1).all() or not np.isfinite(ens_k3).all():
        stop("ensemble prediction contains non-finite values")

    support = [int(np.count_nonzero(to_group(y_true) == cls)) for cls in range(3)]

    print_provenance(runs)
    print(f"True group support: Reduced n={support[0]}, Mildly reduced n={support[1]}, Normal n={support[2]}")
    print("Rates are percentages. Bias is mean predicted EF minus true EF in that true group.")
    print("")
    print(
        "run                      | MAE   RMSE  R2    | acc  mF1  | "
        "R rec/prec/bias | M rec/prec/bias | N rec/prec/bias | "
        "EF<40 TP/FP sens/spec | EF<50 TP/FP sens/spec"
    )
    for run in runs:
        print_result_row(run["label"], all_metrics(y_true, run["y_pred"]))
    print_result_row("R3D+Swin k1 (1/2)", all_metrics(y_true, ens_k1))
    print_result_row("R3D+Swin k3 (1/2)", all_metrics(y_true, ens_k3))
    print("")
    print("Both ensembles are computed in memory only and are not saved. These are descriptive paired VAL results, not deployed models.")


if __name__ == "__main__":
    main()
