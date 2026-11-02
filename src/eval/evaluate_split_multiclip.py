"""
Evaluate ONE checkpoint on the VAL (default) or TEST split with K=3 evenly spaced clips per
video, averaged into one prediction per video, and save everything into a NEW folder.
Inference only: no training, no changes to checkpoints, and existing results are never
overwritten. evaluate_split.py is untouched; this is the multi-clip variant of it.

Sampling design (fixed, not searched): 3 evenly spaced windows per video.
  starts = j * (total_frames - span) // 2 for j in {0, 1, 2}, span = num_frames * frame_stride.
  The middle window (j=1) is exactly the single centred clip the reference evaluator uses,
  so each multi-clip run is directly comparable with its k1 counterpart. Videos with
  total_frames <= span are loop-padded exactly as the dataset does (all 3 windows identical
  there). Predictions (EF, EDV, ESV) are averaged over the 3 clips.

Everything else is identical to evaluate_split.py:
  * --num-workers/--batch-size are not needed here (no DataLoader; one video's 3 clips per
    forward pass, which is <= the reference batch of 4)
  * checkpoints load with weights_only=True only; if that fails the script stops and shows the error
  * normalisation: Swin must have saved values in the checkpoint (no guessing); R3D uses the saved
    values if present, otherwise the constants from evaluate_all.py, and says which one it used
  * checkpoint settings that exist (num_frames, frame_stride, ef_thresholds) are compared with the
    evaluation config; any mismatch stops the run; missing settings are reported as provenance gaps
  * filenames come from the dataset CSV order, are checked for count and uniqueness, and are
    saved as a plain Unicode array plus video_ids.txt
  * fixed three-class reference thresholds [40, 50]; undefined metrics are stored as null / printed n/a

Save as src/eval/evaluate_split_multiclip.py and run from echo-ef-project/ (PowerShell, one line):

    python -m src.eval.evaluate_split_multiclip --config configs/base.yaml --checkpoint checkpoints\r3d18_best.pt --arch r3d18 --split VAL --precision fp32 --tag val_r3d18_baseline_fp32_k3
"""
import argparse
import hashlib
import json
import math
import platform
import re
from pathlib import Path

import numpy as np

CLASS_NAMES = ["Reduced", "Mildly reduced", "Normal"]
THRESHOLDS = [40, 50]
EXPECTED_N = {"VAL": 1288, "TEST": 1277}   # verified split sizes of the current FileList.csv
R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
NUM_CLIPS = 3
SAMPLING = ("3 evenly spaced clips per video: starts = j*(total_frames - span)//2 for j in {0,1,2}, "
            "span = num_frames*frame_stride (the middle window equals the single centred reference clip); "
            "videos with total_frames <= span are loop-padded so all 3 windows are identical there; "
            "no temporal augmentation; EF/EDV/ESV predictions averaged over the 3 clips")


# ----------------------------------------------------------------------------- small helpers
def fmt(x, spec=".3f"):
    """Format a number, or 'n/a' when it is None / NaN / infinite."""
    if x is None:
        return "n/a"
    try:
        if not math.isfinite(float(x)):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return format(float(x), spec)


def finite_or_none(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sha256_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def validate_tag(tag):
    if not isinstance(tag, str) or not TAG_RE.match(tag) or tag.endswith(".") or ".." in tag:
        raise ValueError(f"--tag must be a plain folder name (letters, digits, '.', '_', '-'; "
                         f"max 100 chars, no slashes or spaces), got {tag!r}")
    return tag


def validate_norm(mean, std, label):
    """Return (mean, std) as lists of three finite floats with positive std, or raise ValueError."""
    if mean is None or std is None:
        raise ValueError(f"{label}: normalisation values are missing")
    try:
        m = [float(x) for x in list(mean)]
        s = [float(x) for x in list(std)]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label}: normalisation values are not numeric ({e})") from e
    if len(m) != 3 or len(s) != 3:
        raise ValueError(f"{label}: expected 3 channel values for mean and std, got {len(m)} and {len(s)}")
    if not all(math.isfinite(x) for x in m + s):
        raise ValueError(f"{label}: normalisation contains non-finite values")
    if not all(x > 0 for x in s):
        raise ValueError(f"{label}: every std must be > 0, got {s}")
    return m, s


def multiclip_indices(total_frames, num_frames, frame_stride, k=NUM_CLIPS):
    """Evenly spaced window start indices, matching the dataset's evaluation sampling.

    Returns an array of shape (k, num_frames). For total_frames <= span the dataset loop-pads
    a single window, so all k windows are identical to it. Otherwise starts are
    j * max_start // (k - 1), i.e. first / centred / last; the centred window equals the
    reference evaluator's single centred clip.
    """
    span = num_frames * frame_stride
    if total_frames <= span:
        idx = (np.arange(num_frames) * frame_stride) % total_frames
        return np.stack([idx] * k)
    max_start = total_frames - span
    starts = np.arange(k) * max_start // (k - 1)
    return np.stack([s + np.arange(num_frames) * frame_stride for s in starts])


def find_checkpoint_settings(ckpt):
    """Look for preprocessing settings the checkpoint may have saved. Returns {name: (value, where)}."""
    sources = [("checkpoint (top level)", ckpt)]
    for key in ("config", "cfg"):
        c = ckpt.get(key)
        if isinstance(c, dict):
            sources.append((f"checkpoint['{key}']", c))
            if isinstance(c.get("data"), dict):
                sources.append((f"checkpoint['{key}']['data']", c["data"]))
    found = {}
    for name in ("num_frames", "frame_stride", "ef_thresholds"):
        for where, src in sources:
            if name in src:
                found[name] = (src[name], where)
                break
    return found


def compare_checkpoint_settings(found, data_cfg):
    """Return (rows, mismatches). rows describe what was found; mismatches are hard errors."""
    rows, mismatches = [], []
    for name in ("num_frames", "frame_stride", "ef_thresholds"):
        want = data_cfg.get(name)
        if name not in found:
            rows.append({"setting": name, "checkpoint": None, "evaluation_config": want,
                         "status": "not saved in checkpoint"})
            continue
        have, where = found[name]
        if name == "ef_thresholds":
            same = [float(x) for x in have] == [float(x) for x in want]
        else:
            same = int(have) == int(want)
        rows.append({"setting": name, "checkpoint": have, "evaluation_config": want, "where": where,
                     "status": "match" if same else "MISMATCH"})
        if not same:
            mismatches.append(f"{name}: checkpoint has {have!r} ({where}), evaluation config has {want!r}")
    return rows, mismatches


def resolve_normalisation(arch, ckpt):
    """Return (mean, std, source_text, gaps). Swin must have saved values; nothing is guessed."""
    gaps = []
    saved_m, saved_s = ckpt.get("norm_mean"), ckpt.get("norm_std")
    if arch == "swin3d_t":
        if saved_m is None or saved_s is None:
            raise ValueError("Swin checkpoint has no saved norm_mean/norm_std. Not substituting guessed values. "
                             "Check src/train/train_swin3d.py to see which statistics training used, then "
                             "decide how to proceed.")
        m, s = validate_norm(saved_m, saved_s, "Swin checkpoint normalisation")
        return m, s, "saved in checkpoint", gaps
    if saved_m is not None and saved_s is not None:
        m, s = validate_norm(saved_m, saved_s, "R3D checkpoint normalisation")
        same = np.allclose(m, R3D_MEAN, atol=1e-4) and np.allclose(s, R3D_STD, atol=1e-4)
        note = "saved in checkpoint" + ("" if same else " (differs from the R3D constants in evaluate_all.py)")
        return m, s, note, gaps
    m, s = validate_norm(R3D_MEAN, R3D_STD, "R3D constants")
    gaps.append("R3D normalisation is not stored in the checkpoint; the Kinetics constants from "
                "evaluate_all.py were used and are unverified against training")
    return m, s, "R3D constants from evaluate_all.py (not saved in checkpoint)", gaps


# ----------------------------------------------------------------------------- metrics (pure numpy)
def to_class(ef, thresholds):
    cls = np.zeros(len(ef), dtype=int)
    for t in thresholds:
        cls += (ef >= t).astype(int)
    return cls


def boot_ci(values_fn, n, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [values_fn(rng.integers(0, n, n)) for _ in range(reps)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [finite_or_none(lo), finite_or_none(hi)]


def auc_score(y_bool, score):
    if y_bool.all() or (~y_bool).all():
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return finite_or_none(roc_auc_score(y_bool, score))
    except Exception:  # noqa: BLE001 - sklearn missing or failed: report n/a, never crash
        return None


def compute_metrics(y_true, y_pred, thresholds=THRESHOLDS):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be equal-length, non-empty 1-D arrays")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true / y_pred contain non-finite values")
    n = len(y_true)
    err = y_pred - y_true
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r = None
    if n > 1 and y_true.std() > 0 and y_pred.std() > 0:
        r = finite_or_none(np.corrcoef(y_true, y_pred)[0, 1])
    k = len(thresholds) + 1
    cls_true, cls_pred = to_class(y_true, thresholds), to_class(y_pred, thresholds)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (cls_true, cls_pred), 1)

    groups, f1s = {}, []
    for c in range(k):
        tp = int(cm[c, c])
        support, predicted = int(cm[c].sum()), int(cm[:, c].sum())
        f1 = (2 * tp / (support + predicted)) if (support + predicted) else None
        if f1 is not None:
            f1s.append(f1)
        mask = cls_true == c
        groups[CLASS_NAMES[c] if k == 3 else str(c)] = {
            "n_true": support,
            "n_predicted": predicted,
            "recall": (tp / support) if support else None,
            "precision": (tp / predicted) if predicted else None,
            "f1": f1,
            "mae": finite_or_none(np.mean(np.abs(err[mask]))) if mask.any() else None,
            "mean_signed_error": finite_or_none(np.mean(err[mask])) if mask.any() else None,
        }

    def detect(th):
        pos, pred_pos = y_true < th, y_pred < th
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        return {
            "auc": auc_score(pos, -y_pred),
            "sensitivity": (float((pred_pos & pos).sum()) / n_pos) if n_pos else None,
            "specificity": (float((~pred_pos & ~pos).sum()) / n_neg) if n_neg else None,
            "true_positives": int((pred_pos & pos).sum()),
            "false_positives": int((pred_pos & ~pos).sum()),
            "n_positive": n_pos,
        }

    bias = float(err.mean())
    sd = float(err.std(ddof=1)) if n > 1 else None
    return {
        "n": int(n),
        "mae": float(np.mean(np.abs(err))),
        "mae_ci95": boot_ci(lambda i: float(np.mean(np.abs(err[i]))), n),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_ci95": boot_ci(lambda i: float(np.sqrt(np.mean(err[i] ** 2))), n),
        "r2_coefficient_of_determination": (1.0 - sse / sst) if sst > 0 else None,
        "pearson_r": r,
        "pearson_r_squared": (r ** 2) if r is not None else None,
        "target_variance_sst_over_n": sst / n,
        "ef_lt_50": detect(50),
        "ef_lt_40": detect(40),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_f1_groups_used": len(f1s),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "groups": groups,
        "bland_altman": {
            "bias": bias,
            "loa_low": (bias - 1.96 * sd) if sd is not None else None,
            "loa_high": (bias + 1.96 * sd) if sd is not None else None,
        },
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="new folder name under results/eval_split/")
    ap.add_argument("--arch", default="r3d18", choices=["r3d18", "swin3d_t"])
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    args = ap.parse_args()

    # ---- cheap argument checks first (nothing heavy has run yet)
    try:
        validate_tag(args.tag)
    except ValueError as e:
        raise SystemExit(f"STOPPED: {e}")
    out_dir = Path("results") / "eval_split" / args.tag
    if out_dir.exists():
        raise SystemExit(f"STOPPED: {out_dir} already exists. Choose a new --tag (results are never overwritten).")

    import torch
    import yaml

    from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
    from src.models.r3d_multitask import R3DMultiTask

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    if [float(x) for x in data_cfg["ef_thresholds"]] != [float(x) for x in THRESHOLDS]:
        raise SystemExit(f"STOPPED: this reference evaluator is fixed to thresholds {THRESHOLDS}, "
                         f"config has {data_cfg['ef_thresholds']}")
    if args.split == "TEST":
        print("NOTE: TEST has already been seen. Use it only for a frozen, final evaluation.")

    # ---- checkpoint: safe loading only
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"STOPPED: torch.load(weights_only=True) failed for {args.checkpoint}:\n"
                         f"{type(e).__name__}: {e}\nNo unsafe fallback is provided. Paste this error back for review.")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit("STOPPED: checkpoint is not a dict with 'model_state_dict'")
    print(f"Loaded {args.checkpoint}: epoch {ckpt.get('epoch')}, val_mae recorded at training = {ckpt.get('val_mae')}")

    # ---- preprocessing provenance: compare what the checkpoint saved with the evaluation config
    found = find_checkpoint_settings(ckpt)
    setting_rows, mismatches = compare_checkpoint_settings(found, data_cfg)
    if mismatches:
        raise SystemExit("STOPPED: checkpoint settings disagree with the evaluation config:\n  " + "\n  ".join(mismatches))
    gaps = [f"{r['setting']}: not saved in checkpoint, so it could not be checked against the evaluation config"
            for r in setting_rows if r["status"] == "not saved in checkpoint"]
    try:
        mean, std, norm_source, norm_gaps = resolve_normalisation(args.arch, ckpt)
    except ValueError as e:
        raise SystemExit(f"STOPPED: {e}")
    gaps += norm_gaps
    gaps.append("training config, seed, dataset/CSV version and code version at training time are not "
                "verifiable from the checkpoint alone")

    # ---- model
    if args.arch == "swin3d_t":
        from src.models.swin3d_multitask import SwinMultiTask
        model = SwinMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    else:
        model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    use_amp = args.precision == "amp" and device.type == "cuda"
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    print(f"arch={args.arch} split={args.split} device={device.type} precision={args.precision} "
          f"(autocast={use_amp}) clips_per_video={NUM_CLIPS} (one video per forward pass)")
    print(f"normalisation [{norm_source}] mean={mean} std={std}")
    for gap in gaps:
        print(f"  provenance gap: {gap}")

    # ---- data
    ds = EchoNetDataset(
        cache_dir=data_cfg["cache_dir"], file_list_csv=data_cfg["file_list_csv"], split=args.split,
        num_frames=data_cfg["num_frames"], frame_stride=data_cfg["frame_stride"],
        ef_thresholds=THRESHOLDS, augment=AugmentConfig(temporal=False), seed=cfg["seed"],
    )
    if len(ds) != EXPECTED_N[args.split]:
        raise SystemExit(f"STOPPED: {args.split} has {len(ds)} videos, expected {EXPECTED_N[args.split]}")
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    num_frames, frame_stride = int(data_cfg["num_frames"]), int(data_cfg["frame_stride"])
    names, yt, ef_p, edv_t, edv_p, esv_t, esv_p = [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            arr = np.load(ds.cache_dir / f"{Path(row['FileName']).stem}.npy", mmap_mode="r")  # (T, H, W) uint8
            idx = multiclip_indices(arr.shape[0], num_frames, frame_stride)  # (K, num_frames)
            clip = np.asarray(arr[idx], dtype=np.float32) / 255.0  # (K, num_frames, H, W)
            clip_t = torch.from_numpy(clip).unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)  # (K, 3, T, H, W)
            clip_t = (clip_t - mean_t) / std_t
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(clip_t)
            names.append(str(row["FileName"]))
            yt.append(float(row["EF"]))
            ef_p.append(float(out["ef"].float().mean().cpu()))
            edv_t.append(float(row["EDV"]))
            edv_p.append(float(out["edv"].float().mean().cpu()))
            esv_t.append(float(row["ESV"]))
            esv_p.append(float(out["esv"].float().mean().cpu()))
            if (i + 1) % 128 == 0 or (i + 1) == len(ds):
                print(f"  {i + 1}/{len(ds)} videos")
    y_true = np.asarray(yt, dtype=np.float64)
    y_pred = np.asarray(ef_p, dtype=np.float64)
    edv_true = np.asarray(edv_t, dtype=np.float64)
    edv_pred = np.asarray(edv_p, dtype=np.float64)
    esv_true = np.asarray(esv_t, dtype=np.float64)
    esv_pred = np.asarray(esv_p, dtype=np.float64)

    # ---- checks before anything is written
    n_expected = EXPECTED_N[args.split]
    if not (len(names) == len(set(names)) == len(ds) == n_expected):
        raise SystemExit(f"STOPPED: collected {len(names)} names ({len(set(names))} unique); expected {n_expected}")
    if names != ds.df["FileName"].astype(str).tolist():
        raise SystemExit("STOPPED: batch filenames do not match the dataset dataframe order")
    for label, a in (("y_true", y_true), ("y_pred", y_pred), ("edv_true", edv_true), ("edv_pred", edv_pred),
                     ("esv_true", esv_true), ("esv_pred", esv_pred)):
        if a.shape != (n_expected,) or not np.isfinite(a).all():
            raise SystemExit(f"STOPPED: {label} has shape {a.shape} or non-finite values; nothing saved")
    m = compute_metrics(y_true, y_pred, THRESHOLDS)
    m["edv_mae"] = float(np.mean(np.abs(edv_pred - edv_true)))
    m["esv_mae"] = float(np.mean(np.abs(esv_pred - esv_true)))

    csv_path = data_cfg["file_list_csv"]
    meta = {
        "tag": args.tag, "split": args.split, "arch": args.arch,
        "checkpoint": args.checkpoint, "checkpoint_sha256_16": sha256_file(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_mae_recorded": finite_or_none(ckpt.get("val_mae")),
        "checkpoint_arch_field_if_any": ckpt.get("arch") or ckpt.get("model_name"),
        "precision": args.precision, "autocast_active": use_amp, "tf32_disabled": args.precision == "fp32",
        "device": device.type, "num_workers": 0, "clips_per_forward_pass": NUM_CLIPS,
        "clips_per_video": NUM_CLIPS, "sampling": SAMPLING,
        "normalisation_mean": mean, "normalisation_std": std, "normalisation_source": norm_source,
        "checkpoint_vs_config_settings": setting_rows,
        "effective_data_config": {k: data_cfg.get(k) for k in
                                  ("cache_dir", "file_list_csv", "num_frames", "frame_stride", "ef_thresholds")},
        "eval_config_file": args.config,
        "eval_config_sha256_16": sha256_file(args.config),
        "file_list_csv_sha256_16": sha256_file(csv_path),
        "ids_match_csv_order": True, "ids_unique": True,
        "provenance_gaps": gaps,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "note": "3 evenly spaced clips per video, averaged. The middle window equals the single centred clip of evaluate_split.py.",
    }

    # ---- build the JSON text first so a bad value fails before any file exists
    try:
        json_text_no_hash = json.dumps({**meta, **m}, indent=2, allow_nan=False)
    except ValueError as e:
        raise SystemExit(f"STOPPED: metrics contain a non-finite value that cannot be saved ({e}); nothing saved")

    # ---- save into a NEW folder (never overwrites)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"STOPPED: {out_dir} appeared while the run was in progress; choose a new --tag")
    pred_path = out_dir / "predictions.npz"
    np.savez(pred_path, y_true=y_true, y_pred=y_pred, edv_true=edv_true, edv_pred=edv_pred,
             esv_true=esv_true, esv_pred=esv_pred, file_names=np.array(names, dtype=np.str_))
    ids_path = out_dir / "video_ids.txt"
    ids_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    meta["predictions_file"] = str(pred_path)
    meta["predictions_sha256_16"] = sha256_file(pred_path)
    meta["video_ids_sha256_16"] = sha256_file(ids_path)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({**meta, **m}, f, indent=2, allow_nan=False)

    # ---- report
    print(f"\n=== {args.tag} | {args.split} | n={m['n']} | {args.precision} | {NUM_CLIPS} evenly spaced clips (averaged) ===")
    print(f"MAE {fmt(m['mae'])} (95% CI {fmt(m['mae_ci95'][0])}-{fmt(m['mae_ci95'][1])})   RMSE {fmt(m['rmse'])}   "
          f"true R2 {fmt(m['r2_coefficient_of_determination'])}   Pearson r {fmt(m['pearson_r'])}")
    print(f"target variance SST/n = {fmt(m['target_variance_sst_over_n'], '.4f')}")
    recorded = finite_or_none(ckpt.get("val_mae"))
    if args.split == "VAL" and recorded is not None:
        print(f"recorded val_mae at training = {recorded:.4f}   recomputed now = {m['mae']:.4f}   "
              f"difference = {m['mae'] - recorded:+.4f}")
    for key in ("ef_lt_50", "ef_lt_40"):
        d = m[key]
        print(f"{key}: AUC {fmt(d['auc'])}  sens {fmt(d['sensitivity'])}  spec {fmt(d['specificity'])}  "
              f"TP {d['true_positives']}  FP {d['false_positives']}")
    print(f"macro F1 {fmt(m['macro_f1'])} (over {m['macro_f1_groups_used']} groups)")
    print("Confusion matrix (rows true, cols pred):", CLASS_NAMES)
    print(np.array(m["confusion_matrix_rows_true_cols_pred"]))
    for g, d in m["groups"].items():
        print(f"  {g:15s} n={d['n_true']:4d}  recall {fmt(d['recall'])}  precision {fmt(d['precision'])}  "
              f"MAE {fmt(d['mae'], '.2f')}  mean signed error {fmt(d['mean_signed_error'], '+.2f')}")
    ba = m["bland_altman"]
    print(f"Bland-Altman bias {fmt(ba['bias'])}  LoA [{fmt(ba['loa_low'])}, {fmt(ba['loa_high'])}]")
    print(f"checkpoint sha256[:16] = {meta['checkpoint_sha256_16']}   predictions sha256[:16] = {meta['predictions_sha256_16']}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
