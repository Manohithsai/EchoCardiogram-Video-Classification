"""
Read-only equal-weight ensemble analysis for the three saved VAL EF prediction files.

Ensemble prediction = mean(R3D baseline, Swin baseline, R3D LDS), with equal 1/3
weights. True groups and decision thresholds stay fixed at [40, 50]. There is no
weight search, cutoff search, model loading, checkpoint loading, inference,
training, TEST access, or file writing.

Run from the echo-ef-project folder:

    python analyze_val_equal_ensemble.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_N = 1288
EXPECTED_SUPPORT = (156, 133, 999)  # Reduced, Mildly reduced, Normal on current VAL
THRESHOLDS = (40.0, 50.0)

RUN_FOLDERS = (
    ("R3D baseline", "val_r3d18_baseline_fp32_k1"),
    ("Swin baseline", "val_swin3d_t_baseline_fp32_k1"),
    ("R3D LDS", "val_r3d18_lds_fp32_k1"),
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


def load_run(results_root, label, folder):
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
    if int(meta["clips_per_video"]) != 1:
        stop(f"{run_dir} does not record one clip per video")

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
        stop(f"could not safely read {pred_path}: {e}")

    try:
        ids_from_txt = ids_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e:
        stop(f"could not read Unicode IDs from {ids_path}: {e}")
    if ids_from_txt != names:
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
    print("READ-ONLY VAL EQUAL-WEIGHT ENSEMBLE")
    print("Ensemble EF prediction = mean(R3D baseline, Swin baseline, R3D LDS); weights = 1/3 each")
    print("True groups and decision thresholds stay fixed at [40, 50]. No weights or cutoffs are searched.")
    print("No checkpoint, model, training data, or TEST file is read. No files are written.")
    print("")
    print("Artifact provenance and checks:")
    for run in runs:
        meta = run["meta"]
        print(
            f"  {run['label']}: folder={run['run_dir']} | checkpoint_sha={meta['checkpoint_sha256_16']} | "
            f"predictions_sha={run['pred_hash']} | normalisation={meta['normalisation_source']}"
        )
    print("  Passed: n=1288, split=VAL, FP32, one clip, unique IDs, identical ID order, identical labels, saved hashes match")
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
    parser = argparse.ArgumentParser(description="Read-only equal-weight ensemble on saved VAL predictions")
    parser.add_argument("--results-root", default="results/eval_split")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        stop(f"results root not found: {results_root}. Run this from the echo-ef-project folder")

    runs = [load_run(results_root, label, folder) for label, folder in RUN_FOLDERS]
    reference = runs[0]
    for run in runs[1:]:
        if run["names"] != reference["names"]:
            stop(f"{run['label']} video IDs do not match the baseline order")
        if not np.array_equal(run["y_true"], reference["y_true"]):
            stop(f"{run['label']} EF labels do not exactly match the baseline labels")
        for field in ("eval_config_sha256_16", "file_list_csv_sha256_16"):
            if run["meta"][field] != reference["meta"][field]:
                stop(f"{run['label']} {field} does not match the baseline run")

    y_true = reference["y_true"]
    support = tuple(int(np.count_nonzero(to_group(y_true) == i)) for i in range(3))
    if support != EXPECTED_SUPPORT:
        stop(f"VAL group support is {support}, expected {EXPECTED_SUPPORT}")

    ensemble_pred = np.mean(np.stack([run["y_pred"] for run in runs]), axis=0)
    if not np.isfinite(ensemble_pred).all():
        stop("ensemble prediction contains non-finite values")

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
    print_result_row("Equal-weight ensemble", all_metrics(y_true, ensemble_pred))
    print("")
    print("The ensemble is computed in memory only and is not saved. This is an exploratory VAL result, not a deployed model.")


if __name__ == "__main__":
    main()
