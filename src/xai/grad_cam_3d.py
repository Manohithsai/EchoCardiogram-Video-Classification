"""
3D Grad-CAM for R3D-18. Hooks the last residual stage (layer4) to get
class-activation maps over (time, height, width), then weights channels
by the gradient of the EF output w.r.t. those activations -- the
standard Grad-CAM recipe, extended to a spatiotemporal feature map
instead of a 2D one.
"""
import torch
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (1, 3, T, H, W). Returns a (T', H', W') importance map
        at the resolution of the hooked layer (coarser than the input,
        since R3D-18 downsamples time and space through the stages)."""
        self.model.zero_grad()
        out = self.model(clip)
        ef_pred = out["ef"]
        ef_pred.backward()

        # Global-average-pool the gradients over space+time to get one
        # importance weight per channel (standard Grad-CAM weighting).
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, T', H', W')
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)  # (T', H', W')
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu()


# =============================================================================
# Swin3D-T attention-flow map (added; everything above is the original code,
# byte-for-byte, so GradCAM3D behaves exactly as before).
#
# WHAT THIS IS: a heuristic "attention flow" map for SwinMultiTask. Start with
# equal relevance on the final tokens (the head mean-pools them), then walk
# backwards through the 12 attention blocks and 3 patch-merging layers. In each
# block half of a token's relevance stays on the residual path, half is sent to
# the tokens it attended to (head-averaged attention probabilities).
# WHAT IT IGNORES: the MLP, the value/projection weights, LayerNorm, and the
# difference between heads. Merging splits relevance equally over the 2x2
# inputs. So this is NOT gradient attribution and NOT validated: matching the
# stock forward pass proves the capture is faithful, not that the map explains
# the prediction. Research visual only.
#
# Capture uses forward hooks only, so the stock forward pass is never replaced.
# Each hook recomputes the block's window attention with the same steps as
# torchvision v0.26.0 (pad, roll, window partition, relative bias, shift mask,
# softmax) and checks its own output against the stock output.
# =============================================================================
import argparse
import math
from pathlib import Path

CAPTURE_TOL = 1e-4      # my choice, not measured: max abs diff, recomputed vs stock block output
EF_TOL = 1e-5           # my choice, not measured: EF with hooks vs without hooks
MASS_TOL = 1e-4         # my choice, not measured: relevance mass must stay 1


def _win_part(x, win):
    """(T,H,W) -> (nW, N). Same view/permute as the stock forward, without batch."""
    t, h, w = x.shape
    wt, wh, ww = win
    x = x.reshape(t // wt, wt, h // wh, wh, w // ww, ww)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(-1, wt * wh * ww)


def _win_unpart(x, win, size):
    """(nW, N) -> (T,H,W). Inverse of _win_part."""
    t, h, w = size
    wt, wh, ww = win
    x = x.reshape(t // wt, h // wh, w // ww, wt, wh, ww)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(t, h, w)


def _tv_swin3d():
    from torchvision.models.video import swin_transformer as st
    return st


def _recompute_attention(attn_mod, x):
    """Recompute one ShiftedWindowAttention3d on x (B,T,H,W,C), eval mode.
    Returns (probs (B*nW, heads, N, N), output (B,T,H,W,C), meta dict)."""
    st = _tv_swin3d()
    F = torch.nn.functional
    if attn_mod.training:
        raise RuntimeError("attention flow needs eval mode (dropout off)")
    b, t, h, w, c = x.shape
    window, shift = st._get_window_and_shift_size(list(attn_mod.shift_size), [t, h, w], list(attn_mod.window_size))
    pad = st._compute_pad_size_3d((t, h, w), tuple(window))
    xp = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
    _, tp, hp, wp, _ = xp.shape
    if sum(shift) > 0:
        xp = torch.roll(xp, shifts=(-shift[0], -shift[1], -shift[2]), dims=(1, 2, 3))
    nw = (tp // window[0]) * (hp // window[1]) * (wp // window[2])
    n = window[0] * window[1] * window[2]
    xw = xp.view(b, tp // window[0], window[0], hp // window[1], window[1], wp // window[2], window[2], c)
    xw = xw.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(b * nw, n, c)
    heads = attn_mod.num_heads
    qkv = F.linear(xw, attn_mod.qkv.weight, attn_mod.qkv.bias)
    qkv = qkv.reshape(xw.size(0), n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // heads) ** -0.5
    a = q.matmul(k.transpose(-2, -1)) + attn_mod.get_relative_position_bias(window)
    if sum(shift) > 0:
        mask = st._compute_attention_mask_3d(xw, (tp, hp, wp), tuple(window), tuple(shift))
        a = a.view(b, nw, heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        a = a.view(-1, heads, n, n)
    probs = F.softmax(a, dim=-1)
    o = probs.matmul(v).transpose(1, 2).reshape(xw.size(0), n, c)
    o = F.linear(o, attn_mod.proj.weight, attn_mod.proj.bias)
    o = o.view(b, tp // window[0], hp // window[1], wp // window[2], window[0], window[1], window[2], c)
    o = o.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)
    if sum(shift) > 0:
        o = torch.roll(o, shifts=(shift[0], shift[1], shift[2]), dims=(1, 2, 3))
    o = o[:, :t, :h, :w, :]
    meta = {"window": tuple(window), "shift": tuple(shift), "size": (t, h, w), "pad": tuple(pad)}
    return probs, o, meta


class SwinAttentionFlow:
    """Captures per-block attention with forward hooks, then propagates relevance backwards."""

    def __init__(self, swin_multitask, residual_share=0.5):
        self.model = swin_multitask
        self.feats = swin_multitask.backbone.features
        self.residual_share = residual_share
        self.cap = {}          # id(attn module) -> dict(A (nW,N,N) head-mean, meta, diff)
        self.merge_in = {}     # id(PatchMerging) -> (T,H,W) of its input
        self._hooks = []

    def _attach(self):
        for child in self.feats:
            if isinstance(child, torch.nn.Sequential):
                for blk in child:
                    self._hooks.append(blk.attn.register_forward_hook(self._on_attn))
            else:
                self._hooks.append(child.register_forward_hook(self._on_merge))

    def _detach(self):
        for hk in self._hooks:
            hk.remove()
        self._hooks = []

    def _on_attn(self, mod, inp, out):
        with torch.no_grad():
            probs, o, meta = _recompute_attention(mod, inp[0])
            if meta["pad"] != (0, 0, 0):
                raise RuntimeError(f"attention block needs padding {meta['pad']}; this module only supports pad 0")
            self.cap[id(mod)] = {
                "A": probs.mean(dim=1).detach().cpu().double(),   # (nW, N, N) head-mean (batch must be 1)
                "meta": meta,
                "row_sum_err": float((probs.sum(-1) - 1).abs().max()),
                "diff": float((o - out).abs().max()),
            }

    def _on_merge(self, mod, inp, out):
        t, h, w = inp[0].shape[1:4]
        self.merge_in[id(mod)] = (t, h, w)

    def run(self, clip):
        """clip: normalised (1,3,T,H,W). Returns dict with token relevance and gate numbers."""
        if clip.shape[0] != 1:
            raise ValueError("batch size must be 1")
        self.model.eval()
        self.cap, self.merge_in = {}, {}
        with torch.no_grad():
            stock = self.model(clip)["ef"].float().reshape(-1)       # no hooks yet
            self._attach()
            try:
                hooked = self.model(clip)["ef"].float().reshape(-1)
            finally:
                self._detach()
        rel, masses = self._propagate()
        return {
            "relevance": rel,                                         # (T',H',W') tokens, sums to 1
            "ef_stock": float(stock), "ef_hooked": float(hooked),
            "ef_diff": float((stock - hooked).abs().max()),
            "max_block_diff": max(c["diff"] for c in self.cap.values()),
            "max_row_sum_err": max(c["row_sum_err"] for c in self.cap.values()),
            "n_blocks": len(self.cap), "n_merges": len(self.merge_in),
            "mass_min": min(masses), "mass_max": max(masses),
        }

    def _propagate(self):
        last_blk = [c for c in self.feats if isinstance(c, torch.nn.Sequential)][-1][-1]
        t, h, w = self.cap[id(last_blk.attn)]["meta"]["size"]
        r = torch.full((t, h, w), 1.0 / (t * h * w), dtype=torch.float64)
        masses = []
        for child in reversed(list(self.feats)):
            if isinstance(child, torch.nn.Sequential):
                for blk in reversed(list(child)):
                    r = block_backward(r, self.cap[id(blk.attn)], self.residual_share)
                    masses.append(float(r.sum()))
            else:
                r = merge_backward(r, self.merge_in[id(child)])
                masses.append(float(r.sum()))
        return r, masses


def block_backward(r, cap, share=0.5):
    """r: (T,H,W) relevance at block output. cap: {'A': (nW,N,N), 'meta'}.
    Residual path keeps `share`; the rest goes to attended tokens: r_in[j] += sum_i A[i,j] r[i]
    inside each window, in the rolled/partitioned layout the forward pass used."""
    meta = cap["meta"]
    win, shift, size = meta["window"], meta["shift"], meta["size"]
    x = r
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(-shift[0], -shift[1], -shift[2]), dims=(0, 1, 2))
    xw = _win_part(x, win)                                   # (nW, N)
    yw = torch.einsum("wij,wi->wj", cap["A"].to(xw.dtype), xw)
    y = _win_unpart(yw, win, size)
    if sum(shift) > 0:
        y = torch.roll(y, shifts=(shift[0], shift[1], shift[2]), dims=(0, 1, 2))
    return share * r + (1.0 - share) * y


def merge_backward(r, in_size):
    """Undo PatchMerging (2x2 spatial). r: (T,Ho,Wo). Each cell's relevance is split equally
    over its REAL inputs; padded positions (7 -> 8) get none, so no mass is lost."""
    t, h, w = in_size
    ho, wo = r.shape[1], r.shape[2]
    real = torch.zeros(2 * ho, 2 * wo, dtype=r.dtype)
    real[:h, :w] = 1.0
    count = real.reshape(ho, 2, wo, 2).sum(dim=(1, 3))      # real inputs per cell
    share = r / count.unsqueeze(0)
    up = share.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    return up[:, :h, :w]


def token_map_to_clip(rel, patch=(2, 4, 4)):
    """(T',H',W') tokens -> (T,H,W) pixels by repetition (PatchEmbed3d patch [2,4,4])."""
    out = rel
    for d, p in enumerate(patch):
        out = out.repeat_interleave(p, dim=d)
    return out


# ----------------------------------------------------------------------------- self test
def _naive_first_token_expected(size, win, shift, share):
    """Independent loop check. Every token attends only to the first token of its (rolled) window.
    roll(-s): rolled[q] = orig[(q+s) % n], so original p sits at q=(p-s)%n."""
    t, h, w = size
    exp = [[[0.0] * w for _ in range(h)] for _ in range(t)]
    r = 1.0 / (t * h * w)
    for i in range(t):
        for j in range(h):
            for k in range(w):
                exp[i][j][k] += share * r
                q = [(i - shift[0]) % t, (j - shift[1]) % h, (k - shift[2]) % w]
                q0 = [win[0] * (q[0] // win[0]), win[1] * (q[1] // win[1]), win[2] * (q[2] // win[2])]
                p0 = [(q0[0] + shift[0]) % t, (q0[1] + shift[1]) % h, (q0[2] + shift[2]) % w]
                exp[p0[0]][p0[1]][p0[2]] += (1 - share) * r
    return torch.tensor(exp, dtype=torch.float64)


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    # 1 partition round trip, shifted and not, stage-1 and stage-4 shapes
    for size, win in (((16, 28, 28), (8, 7, 7)), ((16, 4, 4), (8, 4, 4))):
        x = torch.rand(*size, dtype=torch.float64)
        check(f"partition round trip {size}", torch.equal(_win_unpart(_win_part(x, win), win, size), x))

    # 2 identity attention returns the input; uniform attention conserves mass
    size, win = (16, 28, 28), (8, 7, 7)
    n = win[0] * win[1] * win[2]
    nw = (size[0] // win[0]) * (size[1] // win[1]) * (size[2] // win[2])
    r = torch.rand(*size, dtype=torch.float64)
    r = r / r.sum()
    for shift in ((0, 0, 0), (4, 3, 3)):
        cap = {"A": torch.eye(n, dtype=torch.float64).repeat(nw, 1, 1), "meta": {"window": win, "shift": shift, "size": size, "pad": (0, 0, 0)}}
        check(f"identity attention keeps relevance, shift {shift}", torch.allclose(block_backward(r, cap), r, atol=1e-12))
        capu = {"A": torch.full((nw, n, n), 1.0 / n, dtype=torch.float64), "meta": cap["meta"]}
        check(f"uniform attention conserves mass, shift {shift}", abs(float(block_backward(r, capu).sum()) - 1.0) < 1e-9)

    # 3 direction of roll/partition against an independent loop
    for shift in ((0, 0, 0), (4, 3, 3)):
        A = torch.zeros(nw, n, n, dtype=torch.float64)
        A[:, :, 0] = 1.0
        cap = {"A": A, "meta": {"window": win, "shift": shift, "size": size, "pad": (0, 0, 0)}}
        r0 = torch.full(size, 1.0 / (16 * 28 * 28), dtype=torch.float64)
        got = block_backward(r0, cap, 0.5)
        want = _naive_first_token_expected(size, win, shift, 0.5)
        check(f"roll/partition direction vs loop, shift {shift}", torch.allclose(got, want, atol=1e-12),
              f"max diff {float((got - want).abs().max()):.2e}")

    # 4 merging: mass conserved, including 7 -> 4 with padding
    for in_size in ((16, 28, 28), (16, 14, 14), (16, 7, 7)):
        ho = (in_size[1] + 1) // 2
        rr = torch.rand(in_size[0], ho, ho, dtype=torch.float64)
        rr = rr / rr.sum()
        out = merge_backward(rr, in_size)
        check(f"merge backward {in_size[1]}x{in_size[2]} -> {ho}x{ho}", out.shape == in_size and abs(float(out.sum()) - 1) < 1e-12)
    rr = torch.zeros(1, 4, 4, dtype=torch.float64)
    rr[0, 3, 3] = 1.0
    out = merge_backward(rr, (1, 7, 7))
    check("padded corner cell sends all mass to its one real input", abs(float(out[0, 6, 6]) - 1.0) < 1e-12)

    # 5 real torchvision Swin3D-T with random weights (no checkpoint, no data), eval mode
    from src.models.swin3d_multitask import SwinMultiTask
    torch.manual_seed(0)
    model = SwinMultiTask(pretrained=False, dropout_p=0.3).eval()
    clip = torch.randn(1, 3, 32, 112, 112)
    flow = SwinAttentionFlow(model)
    res = flow.run(clip)
    check("12 attention blocks and 3 merges captured", res["n_blocks"] == 12 and res["n_merges"] == 3,
          f"blocks {res['n_blocks']} merges {res['n_merges']}")
    check("recomputed block output equals stock", res["max_block_diff"] < CAPTURE_TOL, f"max abs diff {res['max_block_diff']:.2e}, tol {CAPTURE_TOL} (my choice)")
    check("attention rows sum to 1", res["max_row_sum_err"] < 1e-4, f"max err {res['max_row_sum_err']:.2e}")
    check("EF with hooks equals EF without hooks", res["ef_diff"] < EF_TOL, f"diff {res['ef_diff']:.2e}, tol {EF_TOL} (my choice)")
    check("relevance mass stays 1 end to end", abs(res["mass_min"] - 1) < MASS_TOL and abs(res["mass_max"] - 1) < MASS_TOL,
          f"min {res['mass_min']:.6f} max {res['mass_max']:.6f}")
    check("token map shape (16,28,28)", tuple(res["relevance"].shape) == (16, 28, 28))
    check("hooks removed after run", all(len(b.attn._forward_hooks) == 0 for c in flow.feats if isinstance(c, torch.nn.Sequential) for b in c))
    print("SELFTEST " + ("PASSED" if ok else "FAILED") + "  (random weights, synthetic input; says nothing about the trained model)")
    return ok


# ----------------------------------------------------------------------------- one-VAL-video runner
def run_swin_rollout(args):
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
    from src.eval.evaluate_split import (THRESHOLDS, compare_checkpoint_settings, find_checkpoint_settings,
                                         resolve_normalisation, sha256_file)
    from src.models.swin3d_multitask import SwinMultiTask
    from src.xai.validate_gradcam import get_keyframes

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    torch.backends.cudnn.allow_tf32 = False          # same precision choice as predict_uncertainty
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("arch") != "swin3d_t":
        raise SystemExit(f"STOPPED: checkpoint arch is {ckpt.get('arch')!r}, expected 'swin3d_t'")
    mean, std, norm_source, _gaps = resolve_normalisation("swin3d_t", ckpt)
    rows, mism = compare_checkpoint_settings(find_checkpoint_settings(ckpt), data_cfg)
    print(f"checkpoint sha256[:16] = {sha256_file(args.checkpoint)}  epoch = {ckpt.get('epoch')}  val_mae = {ckpt.get('val_mae')}")
    print(f"normalisation [{norm_source}] mean={mean} std={std}")
    for row in rows:
        print("  setting check:", row)
    if mism:
        raise SystemExit("STOPPED: checkpoint and config disagree: " + "; ".join(mism))

    ds = EchoNetDataset(cache_dir=data_cfg["cache_dir"], file_list_csv=data_cfg["file_list_csv"], split="VAL",
                        num_frames=data_cfg["num_frames"], frame_stride=data_cfg["frame_stride"],
                        ef_thresholds=THRESHOLDS, augment=AugmentConfig(temporal=False), seed=cfg["seed"])
    names = [Path(str(x)).stem for x in ds.df["FileName"]]
    vid = Path(args.video_id).stem
    if vid not in names:
        raise SystemExit(f"STOPPED: {vid} is not in the VAL split (this tool is limited to VAL)")
    item = ds[names.index(vid)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    clip = (item["clip"].unsqueeze(0).to(device) - mean_t) / std_t

    res = SwinAttentionFlow(model).run(clip)
    gates = {
        "recomputed blocks equal stock": res["max_block_diff"] < CAPTURE_TOL,
        "attention rows sum to 1": res["max_row_sum_err"] < 1e-4,
        "EF with hooks equals EF without hooks": res["ef_diff"] < EF_TOL,
        "relevance mass stays 1": abs(res["mass_min"] - 1) < MASS_TOL and abs(res["mass_max"] - 1) < MASS_TOL,
    }
    print(f"EF (stock) = {res['ef_stock']:.4f}   EF (hooked) = {res['ef_hooked']:.4f}   true EF = {float(item['ef']):.2f}")
    print(f"max block diff {res['max_block_diff']:.2e} (tol {CAPTURE_TOL}, my choice)  EF diff {res['ef_diff']:.2e} (tol {EF_TOL}, my choice)")
    for k, v in gates.items():
        print(("PASS  " if v else "FAIL  ") + k)
    if not all(gates.values()):
        raise SystemExit("STOPPED: a gate failed; no picture written")

    pix = token_map_to_clip(res["relevance"]).numpy()                 # (32,112,112)
    pix = pix / pix.max()
    per_frame = pix.sum(axis=(1, 2))
    per_frame = per_frame / per_frame.sum()
    total = np.load(Path(data_cfg["cache_dir"]) / f"{vid}.npy", mmap_mode="r").shape[0]
    idx = ds._sample_clip_indices(total)
    marks = {}
    tr = Path(data_cfg["volume_tracings_csv"])
    if tr.exists():
        for lab, fr in zip(("ED/ES", "ED/ES"), get_keyframes(pd.read_csv(tr), vid)):
            pos = np.where(idx == fr)[0]
            if len(pos):
                marks.setdefault(lab, []).append(int(pos[0]))
    raw = item["clip"][0].numpy()                                     # (32,112,112) in [0,1]
    show = np.linspace(0, raw.shape[0] - 1, 6).round().astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4), gridspec_kw={"height_ratios": [3, 1.6]})
    for a, f in zip(axes[0], show):
        a.imshow(raw[f], cmap="gray", vmin=0, vmax=1)
        a.imshow(pix[f], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        a.set_title(f"clip frame {f}")
        a.axis("off")
    gs = axes[1, 0].get_gridspec()
    for a in axes[1]:
        a.remove()
    ax = fig.add_subplot(gs[1, :])
    ax.plot(per_frame)
    for lab, ps in marks.items():
        for p in ps:
            ax.axvline(p, color="k", ls="--", lw=0.8)
    ax.set_xlabel("clip frame (dashed = annotated ED/ES frames that fall inside this clip)")
    ax.set_ylabel("share of relevance")
    fig.suptitle(f"{vid} VAL  EF pred {res['ef_stock']:.1f}  true {float(item['ef']):.1f}  |  "
                 "Swin attention-flow heuristic, NOT validated, research only", fontsize=11)
    out = Path(args.out) if args.out else Path(cfg["output"]["results_dir"]) / "gradcam" / f"swin_rollout_{vid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}  (video-derived picture: keep local, possible exposure under the Stanford agreement)")


def main():
    ap = argparse.ArgumentParser(description="Swin attention-flow heuristic on ONE VAL clip, or --selftest")
    ap.add_argument("--selftest", action="store_true", help="synthetic checks only; no checkpoint, no data")
    ap.add_argument("--swin-rollout", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/swin3d_t_best.pt")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.swin_rollout:
        if not args.video_id:
            raise SystemExit("--video-id is required")
        run_swin_rollout(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
