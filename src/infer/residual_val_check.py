"""
residual_val_check.py - read-only check of the residual-uncertainty pair on VAL only. Inference only.
No training, no TEST, no new fitting that is kept, nothing written or uploaded. Research model, not a clinical tool.

Question it answers: for the CURRENT local files (checkpoints/r3d18_mcdropout_best.pt as frozen mean model and
checkpoints/r3d18_residual_uncertainty_best.pt as sigma head), what conformal q_hat do they give on VAL, and is it
numerically close to the q_hat saved in results/residual_uncertainty/residual_uncertainty_results.json?
It repeats the original evaluator's steps (src/eval/residual_uncertainty_eval.py) on VAL only: deterministic
eval-mode forward (NOT Monte Carlo), one centred clip, R3D normalisation, no autocast, the model's own logvar clamp
(-6, 6), score = |EF - true| / max(sigma, 1e-3), q_level = min(ceil((n+1)(1-alpha))/n, 1), q_hat = np.quantile(scores,
q_level) with NumPy's default linear interpolation, exactly as the evaluator does.
The recomputed q_hat is a NEW number derived from these files today. A match within tolerance means "numerically
consistent with the saved value". It does not prove the saved value came from these exact files: the saved JSON
records no checkpoint hashes, so that history stays unverified. The SHAs printed here start the record.
Precision note: the original evaluator never set TF32 flags, so PyTorch defaults apply. This script does the same
(it does not disable TF32 the way predict_video.py does). Batch size 4 (the size already verified on this laptop)
instead of the evaluator's 8; BatchNorm is in eval mode, so only tiny float differences are expected.

Run from echo-ef-project, one line:
  python residual_val_check.py
Takes a few minutes (1288 VAL videos).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_residual_uncertainty import R3DResidualUncertainty

R3D_MEAN = [0.43216, 0.394666, 0.37645]
R3D_STD = [0.22803, 0.22145, 0.216989]
MEAN_CKPT = "checkpoints/r3d18_mcdropout_best.pt"
HEAD_CKPT = "checkpoints/r3d18_residual_uncertainty_best.pt"
SAVED = Path("results/residual_uncertainty/residual_uncertainty_results.json")
Q_TOL = 0.01   # my choice, not measured


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    cfg = yaml.safe_load(open("configs/base.yaml", "r"))
    alpha = cfg["uncertainty"]["conformal_alpha"]
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    if abs((1 - alpha) - saved["target_coverage"]) > 1e-9:
        raise SystemExit(f"STOPPED: config alpha {alpha} disagrees with saved target_coverage {saved['target_coverage']}")
    print(f"mean model  {MEAN_CKPT}  sha256[:16]={sha16(MEAN_CKPT)}")
    print(f"sigma head  {HEAD_CKPT}  sha256[:16]={sha16(HEAD_CKPT)}")

    ok = [np.dtype, type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("float16"))]
    try:
        from numpy._core.multiarray import scalar
    except ImportError:
        from numpy.core.multiarray import scalar
    ok.append(scalar)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_ckpt = torch.load(MEAN_CKPT, map_location="cpu", weights_only=True)
    with torch.serialization.safe_globals(ok):
        head_ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=True)
    print(f"head file says mean_checkpoint = {head_ckpt.get('mean_checkpoint')}, epoch {head_ckpt.get('epoch')}")
    if Path(str(head_ckpt.get("mean_checkpoint"))).name != Path(MEAN_CKPT).name:
        raise SystemExit("STOPPED: the head was trained on a different mean checkpoint than the one being used")

    mean_model = R3DMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    mean_model.load_state_dict(mean_ckpt["model_state_dict"])
    mean_model.to(device)
    model = R3DResidualUncertainty(mean_model).to(device)
    model.logvar_head.load_state_dict(head_ckpt["logvar_head_state_dict"])
    model.eval()

    ds = EchoNetDataset(
        cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split="VAL",
        num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
        ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=False), seed=cfg["seed"])
    if len(ds) != 1288:
        raise SystemExit(f"STOPPED: VAL has {len(ds)} videos, expected 1288")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    mean_t = torch.tensor(R3D_MEAN, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(R3D_STD, device=device).view(1, 3, 1, 1, 1)

    efs, sig, true = [], [], []
    with torch.no_grad():
        for batch in loader:
            clip = (batch["clip"].to(device) - mean_t) / std_t
            out = model(clip)
            efs.append(out["ef"].float().cpu().numpy())
            sig.append(torch.exp(0.5 * out["ef_logvar"]).float().cpu().numpy())
            true.append(batch["ef"].numpy())
    ef, sigma, y = np.concatenate(efs), np.concatenate(sig), np.concatenate(true)

    if not (np.isfinite(ef).all() and np.isfinite(sigma).all() and np.isfinite(y).all() and (sigma > 0).all()):
        raise SystemExit("STOPPED: non-finite EF, sigma or label, or sigma <= 0; no q_hat computed")
    scores = np.abs(ef - y) / np.clip(sigma, 1e-3, None)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, q_level))
    print(f"alpha={alpha}  n={n}  q_level={q_level:.5f}")
    print(f"VAL EF MAE of the mean model = {np.abs(ef - y).mean():.4f}  (mean checkpoint recorded val_mae {mean_ckpt.get('val_mae')})")
    print(f"VAL sigma: min {sigma.min():.2f} mean {sigma.mean():.2f} max {sigma.max():.2f} std {sigma.std():.3f}")
    print(f"q_hat recomputed on VAL = {q_hat:.5f}   saved in results JSON = {saved['conformal_q_hat']:.5f}")
    diff = abs(q_hat - saved["conformal_q_hat"])
    print("RESULT:", f"NUMERICALLY CONSISTENT with the saved q_hat (difference {diff:.5f} <= {Q_TOL}). "
          "History of the saved value is still unverified (no hashes were recorded)." if diff <= Q_TOL else
          f"CHECK (difference {diff:.5f} is above {Q_TOL}); the saved q_hat does not match these files. Paste this output.")
    print("Note: VAL also picked the head's best epoch and the mean checkpoint, so this calibration is not independent of selection.")


if __name__ == "__main__":
    main()
