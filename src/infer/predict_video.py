"""
predict_video.py - inference only. One raw EchoNet AVI in, EF / EDV / ESV out, using the SAME steps
as the scored VAL runs (scripts/build_cache.py + EchoNetDataset eval path + evaluate_split.py).

No training, no TEST, no file is written, nothing is uploaded. Research model, not a clinical tool.

Run from the project folder (echo-ef-project), one line, for example:
  python -m src.infer.predict_video --video "data\\EchoNet-Dynamic\\Videos\\<VIDEO>.avi" --checkpoint checkpoints\\r3d18_best.pt --reference results\\eval_split\\val_r3d18_baseline_fp32_k1

--reference (optional) is a saved evaluate_split folder. The script first checks that folder's metrics.json
belongs to THIS checkpoint (SHA-256 prefix, one clip per video, fp32). Only then does it compare the video's
saved EF, EDV and ESV with the new ones (tolerances below are my choice, not measured).
--cache-dir (optional, default data\\cache) lets it also check that the AVI decode gives the same
frames as the cached .npy that the scored run used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.r3d_multitask import R3DMultiTask

# Same constants as evaluate_split.py (R3D / Kinetics values; not stored in the checkpoint).
R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]
FRAME_SIZE = 112     # configs/base.yaml data.frame_size
NUM_FRAMES = 32      # data.num_frames
FRAME_STRIDE = 2     # data.frame_stride
DROPOUT_P = 0.3      # model.dropout_p (irrelevant in eval mode, needed to build the same layers)


def read_video_as_array(video_path, out_size):
    """Copy of scripts/build_cache.py: (T, H, W) uint8 grayscale, INTER_AREA resize."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise SystemExit(f"Zero frames decoded from {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def centred_clip_indices(total_frames):
    """Copy of EchoNetDataset._sample_clip_indices for split != TRAIN."""
    span = NUM_FRAMES * FRAME_STRIDE
    if total_frames <= span:
        return (np.arange(NUM_FRAMES) * FRAME_STRIDE) % total_frames
    max_start = total_frames - span
    start = max_start // 2
    return start + np.arange(NUM_FRAMES) * FRAME_STRIDE


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()[:16]


def predict(model, arr, device):
    idx = centred_clip_indices(arr.shape[0])
    clip = arr[idx].astype(np.float32) / 255.0                       # (T, H, W) in [0, 1]
    clip_t = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)  # (3, T, H, W)
    clip_t = clip_t.unsqueeze(0).to(device)                          # (1, 3, T, H, W)
    mean = torch.tensor(R3D_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(R3D_STD, device=device).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        out = model((clip_t - mean) / std)
    return {k: float(out[k].float().cpu().numpy().reshape(-1)[0]) for k in ("ef", "edv", "esv")}


def ef_group(ef):
    """The project's fixed cutoffs (<40, 40 to <50, >=50). A cutoff group, not a diagnosis."""
    return "below 40" if ef < 40 else ("40 to below 50" if ef < 50 else "50 or above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/r3d18_best.pt")
    ap.add_argument("--reference", default=None, help="saved evaluate_split folder with predictions.npz")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--tol", type=float, default=0.01, help="EF difference accepted as PASS (my choice, not measured)")
    ap.add_argument("--tol-vol", type=float, default=0.05, help="EDV/ESV difference in mL accepted as PASS (my choice, not measured)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:
        raise SystemExit(f"STOPPED: weights_only load failed for {args.checkpoint}: {e}")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit("STOPPED: checkpoint has no model_state_dict (r3d_18_best.pt is a bare state_dict and is not supported here)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = R3DMultiTask(pretrained=False, dropout_p=DROPOUT_P)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"checkpoint {args.checkpoint} sha256[:16]={sha16(args.checkpoint)} epoch={ckpt.get('epoch')} "
          f"val_mae_recorded={ckpt.get('val_mae')} device={device.type} fp32")

    arr = read_video_as_array(video, FRAME_SIZE)
    print(f"decoded {video.name}: {arr.shape[0]} frames at {FRAME_SIZE}x{FRAME_SIZE}")
    res = predict(model, arr, device)
    print(f"EF  = {res['ef']:.3f} %   (project cutoff group: {ef_group(res['ef'])})")
    print(f"EDV = {res['edv']:.2f} mL   ESV = {res['esv']:.2f} mL")
    print("Single centred clip (32 frames, every 2nd frame, a 64-frame span of the video). Research model, "
          "not a clinical tool. This point predictor reports no uncertainty.")

    stem = video.stem
    cache_file = Path(args.cache_dir) / f"{stem}.npy"
    cache_res = None
    if cache_file.is_file():
        cached = np.load(cache_file)
        same = cached.shape == arr.shape and bool(np.array_equal(cached, arr))
        print(f"cache check: {cache_file} frames identical to fresh decode: {same}")
        if not same:
            print("  (decode path differs from the cache the scored run used; the reference check below may not match)")
            cache_res = predict(model, cached, device)
            print(f"  EF from the cached frames = {cache_res['ef']:.3f}")
    else:
        print(f"cache check skipped (no {cache_file})")

    if args.reference:
        ref = Path(args.reference)
        metrics_file = ref / "metrics.json"
        npz = ref / "predictions.npz"
        if not metrics_file.is_file() or not npz.is_file():
            raise SystemExit(f"{args.reference} needs both metrics.json and predictions.npz")
        meta = json.loads(metrics_file.read_text(encoding="utf-8"))
        mine = sha16(args.checkpoint)
        problems = []
        if meta.get("checkpoint_sha256_16") != mine:
            problems.append(f"checkpoint SHA differs (reference {meta.get('checkpoint_sha256_16')}, this {mine})")
        if meta.get("clips_per_video") != 1:
            problems.append(f"reference used {meta.get('clips_per_video')} clips per video, this script uses 1")
        if meta.get("precision") != "fp32":
            problems.append(f"reference precision was {meta.get('precision')}, this script runs fp32")
        if meta.get("arch") != "r3d18":
            problems.append(f"reference arch was {meta.get('arch')}")
        if problems:
            print("reference check NOT VALID, no PASS possible:")
            for p in problems:
                print("  -", p)
            return
        print(f"reference fingerprint OK: {metrics_file.parent.name}, checkpoint sha256[:16]={mine}, "
              f"epoch {meta.get('checkpoint_epoch')}, reference device {meta.get('device')}, batch {meta.get('batch_size')}")
        if meta.get("device") != device.type:
            print(f"  note: reference ran on {meta.get('device')}, this run is on {device.type}; small differences are expected, "
                  "the tolerances were not measured for that case.")
        d = np.load(npz, allow_pickle=False)
        names = [Path(str(n)).stem for n in d["file_names"]]
        if stem not in names:
            print(f"reference: {stem} is not in that VAL run ({len(names)} videos). Pick a video from its video_ids.txt.")
            return
        i = names.index(stem)
        saved = {"ef": float(d["y_pred"][i]), "edv": float(d["edv_pred"][i]), "esv": float(d["esv_pred"][i])}
        print(f"saved:  EF={saved['ef']:.3f}  EDV={saved['edv']:.2f}  ESV={saved['esv']:.2f}")
        print(f"new:    EF={res['ef']:.3f}  EDV={res['edv']:.2f}  ESV={res['esv']:.2f}")
        d_ef = abs(res["ef"] - saved["ef"])
        d_edv = abs(res["edv"] - saved["edv"])
        d_esv = abs(res["esv"] - saved["esv"])
        print(f"abs differences: EF {d_ef:.5f}  EDV {d_edv:.5f}  ESV {d_esv:.5f}")
        ok = d_ef <= args.tol and d_edv <= args.tol_vol and d_esv <= args.tol_vol
        print("RESULT:", "PASS for EF, EDV and ESV on this one video (reproduces the saved VAL run)" if ok else
              f"CHECK (above tolerance EF {args.tol} / volumes {args.tol_vol} mL); do not trust yet, paste this output")

if __name__ == "__main__":
    main()
