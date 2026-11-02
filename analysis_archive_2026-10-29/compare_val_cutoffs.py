"""
Read-only comparison of candidate EF decision cutoffs on saved VAL predictions.

This script does not load checkpoints or models, does not train, does not read TEST,
and does not write or modify any project files. It reads only the three saved
results/eval_split prediction folders listed below.

Run from the echo-ef-project folder:

    python compare_val_cutoffs.py

True EF groups stay fixed at the clinical reference thresholds [40, 50].
Only the cutoffs applied to the model's predicted EF are varied. The continuous
MAE/RMSE/R2 results do not change under these candidate decision cutoffs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_N = 1288
TRUE_REDUCED_CUTOFF = 40.0
TRUE_NORMAL_CUTOFF = 50.0
EXPECTED_SUPPORT = (156, 133, 999)  # current VAL split: Reduced, Mildly reduced, Normal

# Small declared grid. (40, 50) is the original fixed reference row.
CANDIDATE_REDUCED_CUTOFFS = (38.0, 40.0, 43.0, 46.0)
CANDIDATE_NORMAL_CUTOFFS = (48.0, 50.0, 53.0, 56.0)

RUN_SPECS = [
    {
        "label": "R3D baseline",
        "folder": "val_r3d18_baseline_fp32_k1",
    },
    {
        "label": "Swin baseline",
        "folder": "val_swin3d_t_baseline_fp32_k1",
        "alternate_folder": "val_swin3d_t_baseline_fp32_k1",
    },
    {
        "label": "R3D LDS",
        "folder": "val_r3d18_lds_fp32_k1",
    },
]

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

GROUP_NAMES = ("Reduced", "Mildly reduced", "Normal")


def stop(message):
    raise SystemExit(f"STOPPED: {message}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_1d(name, value, expected_n):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape != (expected_n,):
        stop(f"{name} must be a 1-D array with shape ({expected_n},), got {arr.shape}")
    if not np.isfinite(arr.astype(np.float64)).all():
        stop(f"{name} contains non-finite values")
    return arr.astype(np.float64, copy=False)


def true_group(y_true):
    cls = np.zeros(len(y_true), dtype=int)
    cls += y_true >= TRUE_REDUCED_CUTOFF
    cls += y_true >= TRUE_NORMAL_CUTOFF
    return cls


def predicted_group(y_pred, reduced_cutoff, normal_cutoff):
    cls = np.zeros(len(y_pred), dtype=int)
    cls += y_pred >= reduced_cutoff
    cls += y_pred >= normal_cutoff
    return cls


def safe_div(num, den):
    return (num / den) if den else None


def fmt_rate(x):
    return "n/a" if x is None else f"{100.0 * x:5.1f}"


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = None if sst <= 0 else float(1.0 - np.sum(err ** 2) / sst)
    return mae, rmse, r2


def resolve_run_folder(results_root, spec):
    primary = results_root / spec["folder"]
    if primary.is_dir():
        return primary, None
    alternate_name = spec.get("alternate_folder")
    if alternate_name:
        alternate = results_root / alternate_name
        if alternate.is_dir():
            note = f"{spec['label']}: {spec['folder']} not found; using explicit duplicate {alternate_name}"
            return alternate, note
    expected = spec["folder"] + (f" or {alternate_name}" if alternate_name else "")
    stop(f"{spec['label']} folder not found under {results_root}. Expected {expected}")


def load_run(results_root, spec):
    run_dir, note = resolve_run_folder(results_root, spec)
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

    missing = [k for k in REQUIRED_META_FIELDS if k not in meta]
    if missing:
        stop(f"{metrics_path} is missing required fields: {missing}")
    if meta["split"] != "VAL":
        stop(f"{run_dir} is split={meta['split']!r}, expected VAL")
    if int(meta["n"]) != EXPECTED_N:
        stop(f"{run_dir} has n={meta['n']}, expected {EXPECTED_N}")
    if meta["precision"] != "fp32" or meta["autocast_active"] is not False:
        stop(f"{run_dir} is not a saved FP32 run")
    if int(meta["clips_per_video"]) != 1:
        stop(f"{run_dir} does not record one clip per video")
    if meta["tag"] != run_dir.name:
        stop(f"{run_dir} tag mismatch: metrics tag is {meta['tag']!r}")

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
            y_true = finite_1d(f"{run_dir}/y_true", z["y_true"], EXPECTED_N)
            y_pred = finite_1d(f"{run_dir}/y_pred", z["y_pred"], EXPECTED_N)
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
        "label": spec["label"],
        "run_dir": run_dir,
        "note": note,
        "meta": meta,
        "y_true": y_true,
        "y_pred": y_pred,
        "names": names,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pred_hash": pred_hash,
    }


def binary_counts(y_true, y_pred, truth_cutoff, prediction_cutoff):
    truth_pos = y_true < truth_cutoff
    pred_pos = y_pred < prediction_cutoff
    tp = int(np.count_nonzero(truth_pos & pred_pos))
    fp = int(np.count_nonzero(~truth_pos & pred_pos))
    n_pos = int(np.count_nonzero(truth_pos))
    n_neg = int(np.count_nonzero(~truth_pos))
    return {
        "tp": tp,
        "fp": fp,
        "sensitivity": safe_div(tp, n_pos),
        "specificity": safe_div(n_neg - fp, n_neg),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def cutoff_metrics(y_true, y_pred, reduced_cutoff, normal_cutoff):
    yt_cls = true_group(y_true)
    yp_cls = predicted_group(y_pred, reduced_cutoff, normal_cutoff)
    correct = yt_cls == yp_cls
    group_rows = []
    f1_values = []
    for cls in range(3):
        true_mask = yt_cls == cls
        pred_mask = yp_cls == cls
        support = int(np.count_nonzero(true_mask))
        predicted = int(np.count_nonzero(pred_mask))
        tp = int(np.count_nonzero(true_mask & pred_mask))
        false_flags = predicted - tp
        recall = safe_div(tp, support)
        precision = safe_div(tp, predicted)
        f1_den = support + predicted
        f1 = safe_div(2 * tp, f1_den)
        if f1 is not None:
            f1_values.append(f1)
        group_rows.append({
            "support": support,
            "recall": recall,
            "precision": precision,
            "false_flags": false_flags,
        })
    return {
        "accuracy": float(np.count_nonzero(correct) / len(y_true)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "groups": group_rows,
        "ef_lt_40": binary_counts(y_true, y_pred, TRUE_REDUCED_CUTOFF, reduced_cutoff),
        "ef_lt_50": binary_counts(y_true, y_pred, TRUE_NORMAL_CUTOFF, normal_cutoff),
    }


def print_provenance(runs):
    print("READ-ONLY VAL CUTOFF COMPARISON")
    print("True groups are fixed: Reduced EF<40, Mildly reduced 40<=EF<50, Normal EF>=50")
    print("Candidate predicted-EF cutoffs: reduced in [38, 40, 43, 46], normal in [48, 50, 53, 56]")
    print("No checkpoint, model, training data, or TEST file is read. No files are written.")
    print("")
    print("Artifact provenance and checks:")
    for run in runs:
        meta = run["meta"]
        if run["note"]:
            print(f"  NOTE: {run['note']}")
        print(
            f"  {run['label']}: folder={run['run_dir']} | checkpoint_sha={meta['checkpoint_sha256_16']} | "
            f"predictions_sha={run['pred_hash']} | normalisation={meta['normalisation_source']}"
        )
    print("  Passed: n=1288, split=VAL, FP32, one clip, unique IDs, identical ID order, identical labels, saved hashes match")
    print("")


def print_run_table(run):
    meta = run["meta"]
    print(
        f"=== {run['label']} | fixed continuous metrics: MAE {run['mae']:.3f}, "
        f"RMSE {run['rmse']:.3f}, true R2 {run['r2']:.3f} ==="
    )
    print(
        "cutR cutN | acc  mF1  | R rec/prec/FF | M rec/prec/FF | N rec/prec/FF | "
        "EF<40 TP/FP sens/spec | EF<50 TP/FP sens/spec"
    )
    for reduced_cutoff in CANDIDATE_REDUCED_CUTOFFS:
        for normal_cutoff in CANDIDATE_NORMAL_CUTOFFS:
            if reduced_cutoff >= normal_cutoff:
                stop(f"invalid candidate cutoffs: {reduced_cutoff}, {normal_cutoff}")
            m = cutoff_metrics(run["y_true"], run["y_pred"], reduced_cutoff, normal_cutoff)
            g = m["groups"]
            d40 = m["ef_lt_40"]
            d50 = m["ef_lt_50"]
            print(
                f"{reduced_cutoff:4.0f} {normal_cutoff:4.0f} | "
                f"{m['accuracy']:.3f} {m['macro_f1']:.3f} | "
                f"{fmt_rate(g[0]['recall'])}/{fmt_rate(g[0]['precision'])}/{g[0]['false_flags']:3d} | "
                f"{fmt_rate(g[1]['recall'])}/{fmt_rate(g[1]['precision'])}/{g[1]['false_flags']:3d} | "
                f"{fmt_rate(g[2]['recall'])}/{fmt_rate(g[2]['precision'])}/{g[2]['false_flags']:3d} | "
                f"{d40['tp']:3d}/{d40['fp']:3d} {fmt_rate(d40['sensitivity'])}/{fmt_rate(d40['specificity'])} | "
                f"{d50['tp']:3d}/{d50['fp']:3d} {fmt_rate(d50['sensitivity'])}/{fmt_rate(d50['specificity'])}"
            )
    print("")


def main():
    parser = argparse.ArgumentParser(description="Read-only VAL decision-cutoff comparison")
    parser.add_argument("--results-root", default="results/eval_split",
                        help="folder containing the saved evaluate_split run folders")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        stop(f"results root not found: {results_root}. Run this from the echo-ef-project folder")

    runs = [load_run(results_root, spec) for spec in RUN_SPECS]

    reference = runs[0]
    for run in runs[1:]:
        if run["names"] != reference["names"]:
            stop(f"{run['label']} video IDs do not match the baseline order")
        if not np.array_equal(run["y_true"], reference["y_true"]):
            stop(f"{run['label']} EF labels do not exactly match the baseline labels")
        for field in ("eval_config_sha256_16", "file_list_csv_sha256_16"):
            if run["meta"][field] != reference["meta"][field]:
                stop(f"{run['label']} {field} does not match the baseline run")

    support = tuple(int(np.count_nonzero(true_group(reference["y_true"]) == i)) for i in range(3))
    if support != EXPECTED_SUPPORT:
        stop(f"VAL group support is {support}, expected {EXPECTED_SUPPORT}")

    print_provenance(runs)
    print(f"True group support: Reduced n={support[0]}, Mildly reduced n={support[1]}, Normal n={support[2]}")
    print("Rates are percentages. FF means clips falsely assigned to that predicted group.")
    print("")
    for run in runs:
        print_run_table(run)
    print("No candidate is selected as best here. The fixed reference row is cutR=40, cutN=50.")
    print("Other rows are VAL operating-point options only; they do not change MAE/RMSE/R2 and are not clinical thresholds.")


if __name__ == "__main__":
    main()
