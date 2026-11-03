"""
predict_uncertainty.py - inference only. EF with a validation-calibrated range, using the frozen mean model
checkpoints/r3d18_mcdropout_best.pt plus the sigma head checkpoints/r3d18_residual_uncertainty_best.pt.
No training, no TEST, no upload. Research model, not a clinical tool. This is a different model pair from
predict_video.py (r3d18_best.pt); do not compare the two as if they were the same predictor.

Two modes, both run from echo-ef-project:

  1) Calibrate once (VAL only, 1288 videos, a few minutes). Writes ONE new file with the full-precision q_hat,
     the SHA-256 of both checkpoints, alpha, n, settings and sigma quantiles. It refuses to overwrite.
       python -m src.infer.predict_uncertainty --calibrate
     It then runs 3 VAL videos (first, middle, last in the file list) from the raw AVI and compares them with the
     batch values from the same run, as an end-to-end check of the single-video path.

  2) Predict one video. It refuses to run unless both checkpoints have exactly the SHAs stored in the calibration file.
       python -m src.infer.predict_uncertainty --video data\\EchoNet-Dynamic\\Videos\\<ID>.avi

Method (same as src/eval/residual_uncertainty_eval.py): deterministic eval-mode forward (not Monte Carlo), one centred
32-frame clip with stride 2, R3D normalisation, no autocast, TF32 off, sigma = exp(0.5 * logvar) with the model's clamp (-6, 6),
range = EF +/- q_hat * sigma, q_hat = the ceil((n+1)(1-alpha))/n quantile of |EF - true| / sigma on VAL. PyTorch default
TF32 settings are kept on purpose, as in the original evaluator and the calibration pass.
The range is a development-time statistic: VAL also chose the head's epoch and the mean checkpoint, and about 91% of TEST
videos fell inside the range while TEST was being used during development. It is wide and is not a clinical
confidence statement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.infer import predict_video as pv
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_residual_uncertainty import R3DResidualUncertainty

MEAN_CKPT = "checkpoints/r3d18_mcdropout_best.pt"
HEAD_CKPT = "checkpoints/r3d18_residual_uncertainty_best.pt"
CAL_FILE = Path("results/uncertainty_calibration/residual_val_calibration.json")
CAL_TOL = 1e-3   # my choice, not measured
SOURCE_FILES = ["src/infer/predict_uncertainty.py", "src/infer/predict_video.py", "src/models/r3d_residual_uncertainty.py",
                "src/models/r3d_multitask.py", "src/data/echonet_dataset.py", "src/eval/residual_uncertainty_eval.py"]


def sha_full(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def runtime_info(device):
    info = {"torch_version": torch.__version__, "numpy_version": np.__version__, "device": device.type,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def load_pair(device):
    try:
        from numpy._core.multiarray import scalar
    except ImportError:
        from numpy.core.multiarray import scalar
    ok = [scalar, np.dtype] + [type(np.dtype(n)) for n in ("float64", "float32", "float16")]
    mean_ckpt = torch.load(MEAN_CKPT, map_location="cpu", weights_only=True)
    with torch.serialization.safe_globals(ok):
        head_ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=True)
    if Path(str(head_ckpt.get("mean_checkpoint"))).name != Path(MEAN_CKPT).name:
        raise SystemExit("STOPPED: the head was trained on a different mean checkpoint")
    mean_model = R3DMultiTask(pretrained=False, dropout_p=pv.DROPOUT_P)
    mean_model.load_state_dict(mean_ckpt["model_state_dict"])
    mean_model.to(device)
    model = R3DResidualUncertainty(mean_model).to(device)
    model.logvar_head.load_state_dict(head_ckpt["logvar_head_state_dict"])
    model.eval()
    return model, mean_ckpt, head_ckpt


def predict_one(model, arr, device):
    """arr: (T, H, W) uint8 as build_cache makes it. Returns (EF, sigma) from the deterministic forward."""
    idx = pv.centred_clip_indices(arr.shape[0])
    clip = arr[idx].astype(np.float32) / 255.0
    clip_t = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0).to(device)
    mean = torch.tensor(pv.R3D_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(pv.R3D_STD, device=device).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        out = model((clip_t - mean) / std)
    ef = float(out["ef"].float().cpu().numpy().reshape(-1)[0])
    sigma = float(torch.exp(0.5 * out["ef_logvar"]).float().cpu().numpy().reshape(-1)[0])
    if not (np.isfinite(ef) and np.isfinite(sigma) and sigma > 0):
        raise SystemExit(f"STOPPED: bad single prediction (EF {ef}, sigma {sigma})")
    return ef, sigma


def calibrate():
    cfg = yaml.safe_load(open("configs/base.yaml", "r"))
    alpha = cfg["uncertainty"]["conformal_alpha"]
    if not (0 < alpha < 1):
        raise SystemExit(f"STOPPED: alpha {alpha} not in (0, 1)")
    d0 = cfg["data"]
    if (d0["num_frames"], d0["frame_stride"], d0["frame_size"]) != (pv.NUM_FRAMES, pv.FRAME_STRIDE, pv.FRAME_SIZE):
        raise SystemExit("STOPPED: configs/base.yaml data settings differ from the constants the single-video path uses "
                         f"(config {d0['num_frames']}/{d0['frame_stride']}/{d0['frame_size']} vs "
                         f"code {pv.NUM_FRAMES}/{pv.FRAME_STRIDE}/{pv.FRAME_SIZE}); nothing written")
    if CAL_FILE.exists():
        raise SystemExit(f"STOPPED: {CAL_FILE} already exists; not overwriting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean_ckpt, head_ckpt = load_pair(device)
    d = cfg["data"]
    ds = EchoNetDataset(cache_dir=d["cache_dir"], file_list_csv=d["file_list_csv"], split="VAL",
                        num_frames=d["num_frames"], frame_stride=d["frame_stride"], ef_thresholds=d["ef_thresholds"],
                        augment=AugmentConfig(temporal=False), seed=cfg["seed"])
    if len(ds) != 1288:
        raise SystemExit(f"STOPPED: VAL has {len(ds)} videos, expected 1288")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    mean_t = torch.tensor(pv.R3D_MEAN, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(pv.R3D_STD, device=device).view(1, 3, 1, 1, 1)
    efs, sig, true = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model((batch["clip"].to(device) - mean_t) / std_t)
            efs.append(out["ef"].float().cpu().numpy())
            sig.append(torch.exp(0.5 * out["ef_logvar"]).float().cpu().numpy())
            true.append(batch["ef"].numpy())
    ef, sigma, y = np.concatenate(efs), np.concatenate(sig), np.concatenate(true)
    if not (np.isfinite(ef).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise SystemExit("STOPPED: non-finite EF or sigma, or sigma <= 0")
    if not np.isfinite(y).all():
        raise SystemExit("STOPPED: non-finite true EF in VAL")
    scores = np.abs(ef - y) / np.clip(sigma, 1e-3, None)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, q_level))
    if not (np.isfinite(scores).all() and np.isfinite(q_hat) and q_hat > 0):
        raise SystemExit("STOPPED: non-finite scores or q_hat")
    saved = json.loads(Path("results/residual_uncertainty/residual_uncertainty_results.json").read_text(encoding="utf-8"))
    print(f"q_hat on VAL = {q_hat!r}  (older saved JSON value {saved['conformal_q_hat']!r}, difference {abs(q_hat - saved['conformal_q_hat']):.5f})")

    # end-to-end check: raw AVI path vs the batch values of this same run
    worst_ef = worst_sig = 0.0
    for i in (0, n // 2, n - 1):
        stem = Path(str(ds.df.iloc[i]["FileName"])).stem
        arr = pv.read_video_as_array(Path("data/EchoNet-Dynamic/Videos") / f"{stem}.avi", pv.FRAME_SIZE)
        e, s = predict_one(model, arr, device)
        worst_ef = max(worst_ef, abs(e - float(ef[i])))
        worst_sig = max(worst_sig, abs(s - float(sigma[i])))
    print(f"single-video path vs batch, 3 VAL videos: max |dEF| = {worst_ef:.5f}, max |dsigma| = {worst_sig:.5f}")
    if worst_ef > CAL_TOL or worst_sig > CAL_TOL:
        raise SystemExit(f"STOPPED: single-video path differs from the batch path by more than {CAL_TOL}; nothing written")

    art = {
        "what": "residual-uncertainty conformal calibration on VAL, current local files",
        "alpha": alpha, "n_val": n, "q_level": float(q_level), "q_hat": q_hat,
        "mean_checkpoint": MEAN_CKPT, "mean_checkpoint_sha256": sha_full(MEAN_CKPT),
        "mean_checkpoint_epoch": mean_ckpt.get("epoch"), "mean_checkpoint_val_mae": float(mean_ckpt.get("val_mae")),
        "head_checkpoint": HEAD_CKPT, "head_checkpoint_sha256": sha_full(HEAD_CKPT),
        "head_epoch": head_ckpt.get("epoch"), "head_val_loss": float(head_ckpt.get("val_loss")),
        "val_ef_mae": float(np.abs(ef - y).mean()),
        "val_sigma_quantiles_0_to_1_in_100_steps": [float(v) for v in np.quantile(sigma, np.linspace(0, 1, 101))],
        "settings": {"num_frames": pv.NUM_FRAMES, "frame_stride": pv.FRAME_STRIDE, "frame_size": pv.FRAME_SIZE,
                     "mean": pv.R3D_MEAN, "std": pv.R3D_STD, "clips_per_video": 1, "batch_size_in_calibration": 4,
                     "autocast": False, "tf32": "forced off by this script (cudnn and matmul)", "forward": "deterministic eval mode",
                     "logvar_clamp": [-6.0, 6.0], "score_sigma_floor": 1e-3},
        "file_list_csv_sha256": sha_full(d["file_list_csv"]), "config_sha256": sha_full("configs/base.yaml"),
        "source_sha256": {f: sha_full(f) for f in SOURCE_FILES if Path(f).is_file()},
        "runtime": runtime_info(device),
        "caveats": ["VAL also selected the head epoch and the mean checkpoint, so this is not independent of selection",
                    "earlier saved q_hat had no checkpoint hashes; this file starts the hash record",
                    "not a clinical confidence statement"],
    }
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAL_FILE, "x", encoding="utf-8") as f:
        json.dump(art, f, indent=2, allow_nan=False)
    print(f"wrote {CAL_FILE} (new file, local only); mean sha256 {art['mean_checkpoint_sha256']}, head sha256 {art['head_checkpoint_sha256']}")


def diagnose():
    """Bounded check, no file written, no labels used: where does the raw-vs-batch difference come from?"""
    cfg = yaml.safe_load(open("configs/base.yaml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pair(device)
    d = cfg["data"]
    ds = EchoNetDataset(cache_dir=d["cache_dir"], file_list_csv=d["file_list_csv"], split="VAL",
                        num_frames=d["num_frames"], frame_stride=d["frame_stride"], ef_thresholds=d["ef_thresholds"],
                        augment=AugmentConfig(temporal=False), seed=cfg["seed"])
    n = len(ds)
    mean_t = torch.tensor(pv.R3D_MEAN, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(pv.R3D_STD, device=device).view(1, 3, 1, 1, 1)
    print("runtime:", runtime_info(device))

    def run(clips):
        x = torch.stack(clips).to(device)
        with torch.no_grad():
            out = model((x - mean_t) / std_t)
        return (out["ef"].float().cpu().numpy().reshape(-1),
                torch.exp(0.5 * out["ef_logvar"]).float().cpu().numpy().reshape(-1))

    def raw_clip(i):
        stem = Path(str(ds.df.iloc[i]["FileName"])).stem
        arr = pv.read_video_as_array(Path("data/EchoNet-Dynamic/Videos") / f"{stem}.avi", pv.FRAME_SIZE)
        c = torch.from_numpy(arr[pv.centred_clip_indices(arr.shape[0])].astype(np.float32) / 255.0)
        return stem, arr.shape[0], c.unsqueeze(0).repeat(3, 1, 1, 1)

    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    for i in (0, n // 2, n - 1):
        stem, nf, rc = raw_clip(i)
        cc = ds[i]["clip"]
        dc = float((rc - cc).abs().max())
        print(f"\n{stem}  frames={nf}  raw clip vs cached clip: max |diff| = {dc:.6f}  identical={bool(torch.equal(rc, cc))}")
        lo = i // 4 * 4
        group = [ds[j]["clip"] for j in range(lo, min(lo + 4, n))]
        pos = i - lo
        for tf32 in (True, False):
            torch.backends.cudnn.allow_tf32 = tf32
            torch.backends.cuda.matmul.allow_tf32 = tf32
            e_c1, s_c1 = run([cc])
            e_r1, s_r1 = run([rc])
            e_c4, s_c4 = run(group)
            e_cc = run([cc, cc])[0]
            print(f"  tf32={tf32}: cached b1 EF={e_c1[0]:.5f} sig={s_c1[0]:.5f} | "
                  f"raw b1 dEF={abs(e_r1[0]-e_c1[0]):.5f} dsig={abs(s_r1[0]-s_c1[0]):.5f} | "
                  f"cached b4 (as calibrated) dEF={abs(e_c4[pos]-e_c1[0]):.5f} dsig={abs(s_c4[pos]-s_c1[0]):.5f} | "
                  f"repeat in b1 dEF={abs(run([cc])[0][0]-e_c1[0]):.5f}")
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
    print("\nNo file written, no labels used. Differences are reported, not judged against a threshold.")


def predict(video):
    if not CAL_FILE.is_file():
        raise SystemExit(f"STOPPED: {CAL_FILE} not found; run with --calibrate first")
    cal = json.loads(CAL_FILE.read_text(encoding="utf-8"))
    for key, path in (("mean_checkpoint_sha256", MEAN_CKPT), ("head_checkpoint_sha256", HEAD_CKPT),
                      ("config_sha256", "configs/base.yaml")):
        if sha_full(path) != cal[key]:
            raise SystemExit(f"STOPPED: {path} does not match the calibration file; no range reported")
    s = cal["settings"]
    if (s["num_frames"], s["frame_stride"], s["frame_size"], s["mean"], s["std"]) != (pv.NUM_FRAMES, pv.FRAME_STRIDE, pv.FRAME_SIZE, pv.R3D_MEAN, pv.R3D_STD):
        raise SystemExit("STOPPED: preprocessing settings differ from the calibration file")
    notes = []
    for f, h in cal.get("source_sha256", {}).items():
        if f != "src/infer/predict_uncertainty.py" and Path(f).is_file() and sha_full(f) != h:
            notes.append(f"source file changed since calibration: {f}")
    video = Path(video)
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pair(device)
    now, then = runtime_info(device), cal["runtime"]
    for k in now:
        if now[k] != then.get(k):
            notes.append(f"runtime differs from calibration: {k} was {then.get(k)}, now {now[k]}")
    arr = pv.read_video_as_array(video, pv.FRAME_SIZE)
    ef, sigma = predict_one(model, arr, device)
    half = cal["q_hat"] * sigma
    qs = np.array(cal["val_sigma_quantiles_0_to_1_in_100_steps"])
    pct = min(100.0, float(np.searchsorted(qs, sigma, side="right")) * 100.0 / (len(qs) - 1))
    print(f"{video.name}: {arr.shape[0]} frames, device {device.type}")
    print(f"EF = {ef:.2f} %   sigma = {sigma:.2f}   nominal {int(round(100 * (1 - cal['alpha'])))}% range (not a guarantee) = [{ef - half:.1f}, {ef + half:.1f}] (half-width {half:.1f} EF points)")
    print(f"sigma is at roughly the {pct:.0f}th percentile of the VAL sigmas (approximate; higher = larger estimated error).")
    print(f"pair: mean model sha256 {cal['mean_checkpoint_sha256'][:16]}..., head sha256 {cal['head_checkpoint_sha256'][:16]}..., calibrated on {cal['n_val']} VAL videos")
    for n_ in notes:
        print("NOTE:", n_)
    print("Research model, not a clinical tool. The range is a nominal, development-time statistic calibrated on VAL, which also "
          "selected the head epoch and mean checkpoint (so it is selection-dependent); it is wide and not a clinical confidence statement. Single centred clip.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--diagnose", action="store_true")
    a = ap.parse_args()
    # Explicit, for calibration AND prediction: cuDNN TF32 is on by default in PyTorch and changes EF by ~0.01.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.calibrate + a.diagnose + bool(a.video) != 1:
        raise SystemExit("use exactly one of --calibrate, --diagnose or --video <avi>")
    if a.diagnose:
        diagnose()
    elif a.calibrate:
        calibrate()
    else:
        predict(a.video)


if __name__ == "__main__":
    sys.exit(main())
