"""
smoke_predict_videos.py - local smoke test for predict_video.py. Inference only.
No training, no TEST, nothing written or uploaded. Uses predict_video.py unchanged (put both files in the
project folder). Research model, not a clinical tool.

Run from echo-ef-project, one line:
  python -m src.infer.smoke_predict_videos

What it does:
  1. Checks the saved VAL run's metrics.json belongs to checkpoints/r3d18_best.pt (SHA prefix, fp32, 1 clip).
  2. Reads frame counts of all VAL videos from the cache shapes (no decoding), picks the 3 shortest, 2 around
     the median and 3 longest, and says how many have 64 frames or fewer (the loop-pad branch).
  3. For each: decodes the raw AVI, predicts, compares EF, EDV, ESV with the saved run, and checks the AVI
     frames equal the cached frames.
  4. Synthetic check: cuts the shortest cached video to 40 frames to run the loop-pad branch once. It has no
     saved reference, so it only shows the branch runs and gives finite numbers. It is not a validation.
Tolerances (my choice, not measured): EF 0.01, EDV and ESV 0.05 mL.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.infer import predict_video as pv
from src.models.r3d_multitask import R3DMultiTask

CKPT = "checkpoints/r3d18_best.pt"
REF = Path("results/eval_split/val_r3d18_baseline_fp32_k1")
CACHE = Path("data/cache")
VIDEOS = Path("data/EchoNet-Dynamic/Videos")
TOL_EF, TOL_VOL = 0.01, 0.05


def main():
    meta = json.loads((REF / "metrics.json").read_text(encoding="utf-8"))
    mine = pv.sha16(CKPT)
    if (meta.get("checkpoint_sha256_16") != mine or meta.get("clips_per_video") != 1
            or meta.get("precision") != "fp32" or meta.get("arch") != "r3d18"):
        raise SystemExit(f"STOPPED: reference metrics.json does not match {CKPT} (sha {mine}); no PASS possible")
    print(f"reference fingerprint OK ({REF.name}, sha256[:16]={mine})")

    d = np.load(REF / "predictions.npz", allow_pickle=False)
    names = [str(n) for n in d["file_names"]]
    stems = [Path(n).stem for n in names]
    frames = []
    for s in stems:
        frames.append(np.load(CACHE / f"{s}.npy", mmap_mode="r").shape[0])
    frames = np.array(frames)
    order = np.argsort(frames, kind="stable")
    mid = len(order) // 2
    picks = list(order[:3]) + list(order[mid - 1:mid + 1]) + list(order[-3:])
    n_short = int((frames <= pv.NUM_FRAMES * pv.FRAME_STRIDE).sum())
    print(f"VAL videos: {len(frames)}; frames min {frames.min()} / median {int(np.median(frames))} / max {frames.max()}")
    print(f"videos with 64 frames or fewer (loop-pad branch): {n_short} of {len(frames)}")
    print(f"picked frame counts: {[int(frames[i]) for i in picks]}")
    if not any(frames[i] <= 64 for i in picks):
        print("NOTE: none of the picked real videos use the loop-pad branch; that branch is only covered by the synthetic check below.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = R3DMultiTask(pretrained=False, dropout_p=pv.DROPOUT_P)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"device {device.type}, saved run device {meta.get('device')}")

    n_pass = 0
    for i in picks:
        s = stems[i]
        arr = pv.read_video_as_array(VIDEOS / f"{s}.avi", pv.FRAME_SIZE)
        cached = np.load(CACHE / f"{s}.npy")
        same = cached.shape == arr.shape and bool(np.array_equal(cached, arr))
        r = pv.predict(model, arr, device)
        sv = {"ef": float(d["y_pred"][i]), "edv": float(d["edv_pred"][i]), "esv": float(d["esv_pred"][i])}
        df = abs(r["ef"] - sv["ef"]); de = abs(r["edv"] - sv["edv"]); ds = abs(r["esv"] - sv["esv"])
        ok = same and df <= TOL_EF and de <= TOL_VOL and ds <= TOL_VOL
        n_pass += int(ok)
        print(f"{s} frames={arr.shape[0]:4d} cache_same={same} dEF={df:.5f} dEDV={de:.5f} dESV={ds:.5f} -> {'PASS' if ok else 'CHECK'}")
    print(f"real-video result: {n_pass} of {len(picks)} PASS (EF, EDV, ESV and decode)")

    short = np.load(CACHE / f"{stems[order[0]]}.npy")[:40]
    r = pv.predict(model, short, device)
    finite = all(np.isfinite(v) for v in r.values())
    idx = pv.centred_clip_indices(40)
    idx_ok = bool(len(idx) == 32 and idx.min() >= 0 and idx.max() < 40 and np.array_equal(idx, (np.arange(32) * 2) % 40))
    print(f"synthetic 40-frame clip (loop-pad branch): index logic ok={idx_ok}, finite outputs={finite}, "
          f"EF={r['ef']:.2f} (no reference, not a validation)")
    print("Done. Local only; nothing written.")


if __name__ == "__main__":
    sys.exit(main())
