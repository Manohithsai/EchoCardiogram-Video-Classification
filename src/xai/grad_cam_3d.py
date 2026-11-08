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

    use_cam = getattr(args, "swin_gradcam", False)
    res = SwinGradCAM3D(model).run(clip) if use_cam else SwinAttentionFlow(model).run(clip)
    if use_cam:
        print("Grad-CAM final backbone.norm:", res["activation_shape"], "gradient max/mean", res["gradient_abs_max"], res["gradient_abs_mean"])
        print("raw signed CAM range", res["raw_cam_min"], res["raw_cam_max"], "positive fraction", res["positive_fraction"])
        print(f"PASS stock EF {res['ef_stock']:.4f} vs captured {res['ef_hooked']:.4f}; delta {res['ef_diff']:.2e}; finite nonzero gradients")
    if not use_cam:
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

    pix = (res["pixel_cam"] if use_cam else token_map_to_clip(res["relevance"])).numpy()                 # (32,112,112)
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
                 + ("Swin EF Grad-CAM (coarse interpolated map), NOT validated, research only" if use_cam else "Swin attention-flow heuristic, NOT validated, research only"), fontsize=11)
    out = Path(args.out) if args.out else Path(cfg["output"]["results_dir"]) / "gradcam" / f"swin_{'gradcam' if use_cam else 'rollout'}_{vid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        plt.close(fig)
        raise SystemExit(f"STOPPED: {out} exists; choose another --out to preserve prior output")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (video-derived picture: keep local, possible exposure under the Stanford agreement)")


# ----------------------------------------------------------------------------- EF-gradient Swin CAM
class EmptyPositiveCAM(RuntimeError):
    """Valid finite EF gradients, but ReLU leaves no positive attribution."""
    def __init__(self, ef, signed_min, signed_max):
        super().__init__('degenerate CAM: no positive EF contribution; no heatmap written')
        self.ef = ef
        self.signed_min = signed_min
        self.signed_max = signed_max


class SwinGradCAM3D:
    """EF Grad-CAM at final backbone.norm, B,T,H,W,C before spatial/temporal pooling.

    The native lattice is coarse (32x112x112 input -> 16x4x4); interpolation
    is display resampling, not recovered anatomical detail. ReLU keeps positive
    contributions to the EF scalar, not evidence for an EF category.
    """
    def __init__(self, model):
        self.model = model
        self.target = model.backbone.norm

    def run(self, clip):
        if self.model.training or any(m.training for m in self.model.modules()):
            raise RuntimeError('Grad-CAM requires the whole model in eval mode')
        if clip.ndim != 5 or clip.shape[0] != 1 or clip.shape[1] != 3:
            raise ValueError('expected one (1,3,T,H,W) normalised video clip')
        if not torch.isfinite(clip).all():
            raise ValueError('clip contains nonfinite values')
        # Requiring input gradients also works when ALL model parameters are frozen.
        # No parameter .grad buffers are populated: autograd.grad targets activation only.
        with torch.inference_mode(False), torch.enable_grad():
            x = clip.detach().clone().requires_grad_(True)
            with torch.no_grad():
                stock = self.model(x.detach())['ef'].float().reshape(-1)
            captured = []
            handle = self.target.register_forward_hook(lambda module, inputs, output: captured.append(output))
            try:
                out = self.model(x)['ef'].float().reshape(-1)
                if len(captured) != 1:
                    raise RuntimeError('expected exactly one final-norm activation')
                a = captured[0]
                if a.ndim != 5 or a.shape[0] != 1 or not a.requires_grad:
                    raise RuntimeError('expected differentiable channels-last activation B,T,H,W,C')
                g, = torch.autograd.grad(out[0], a, retain_graph=False, create_graph=False)
                if not torch.isfinite(a).all() or not torch.isfinite(g).all():
                    raise RuntimeError('nonfinite activation or EF gradient')
                if float(g.abs().max()) == 0:
                    raise RuntimeError('zero EF gradient at final norm; no usable CAM')
                if not torch.isfinite(out).all() or not torch.isfinite(stock).all():
                    raise RuntimeError('nonfinite EF prediction')
                diff = float((stock - out.detach()).abs().max())
                if diff >= EF_TOL:
                    raise RuntimeError(f'EF changed during capture: {diff:.3g}, tolerance {EF_TOL}')
                signed = (a * g.mean(dim=(1, 2, 3), keepdim=True)).sum(dim=-1)[0].detach()
                if not torch.isfinite(signed).all():
                    raise RuntimeError("nonfinite weighted CAM; no heatmap written")
                positive = signed.clamp_min(0)
                peak = float(positive.max())
                if peak <= 0:
                    raise EmptyPositiveCAM(float(stock[0]), float(signed.min()), float(signed.max()))
                native = positive / peak
                pix = F.interpolate(native[None, None], size=tuple(x.shape[2:]), mode='trilinear', align_corners=False)[0, 0]
                patch_size = tuple(self.model.backbone.patch_embed.proj.stride)
                lattice = tuple((int(s) + int(k) - 1) // int(k) for s, k in zip(x.shape[2:], patch_size))
                rel = F.interpolate(native[None, None], size=lattice, mode='trilinear', align_corners=False)[0, 0]
                return {'ef_stock': float(stock[0]), 'ef_hooked': float(out.detach()[0]),
                        'ef_diff': diff, 'native_cam': native.cpu(), 'pixel_cam': pix.cpu(),
                        'relevance': rel.cpu(), 'activation_shape': tuple(a.shape),
                        'gradient_abs_max': float(g.abs().max()), 'gradient_abs_mean': float(g.abs().mean()),
                        'raw_cam_min': float(signed.min()), 'raw_cam_max': float(signed.max()),
                        'positive_fraction': float((signed > 0).float().mean())}
            finally:
                handle.remove()


def selftest_gradcam():
    from src.models.swin3d_multitask import SwinMultiTask
    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    m = SwinMultiTask(pretrained=False).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 32, 112, 112)
    before = len(m.backbone.norm._forward_hooks)
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    r = SwinGradCAM3D(m).run(x)
    assert r['activation_shape'] == (1, 16, 4, 4, 768)
    assert r['pixel_cam'].shape == (32, 112, 112)
    assert r['relevance'].shape == (16, 28, 28)
    assert r['gradient_abs_max'] > 0 and r['ef_diff'] < EF_TOL
    assert all(p.grad is None for p in m.parameters())
    assert all(torch.equal(v, state[k]) for k, v in m.state_dict().items())
    assert len(m.backbone.norm._forward_hooks) == before
    print('PASS frozen torchvision Swin: finite nonzero EF gradients, channels-last lattice, pixel/patch shapes')
    print('PASS stock EF unchanged; weights/buffers unchanged; no parameter grads; hook removed')
    # Error path must also clean hooks, without inventing a heatmap.
    with torch.no_grad():
        m.head[-1].weight.zero_()
    try:
        SwinGradCAM3D(m).run(x)
        raise AssertionError('zero-gradient case was not rejected')
    except RuntimeError as e:
        assert 'zero EF gradient' in str(e)
    assert len(m.backbone.norm._forward_hooks) == before
    print('PASS zero-gradient CAM rejected, hook removed on failure')
    # Deterministic finite-gradient toy: one positive map and one all-negative signed map.
    # This tests missingness/accounting, not trained-model explanation quality.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.norm = torch.nn.Identity()
            self.backbone.patch_embed = torch.nn.Module()
            self.backbone.patch_embed.proj = torch.nn.Conv3d(3, 3, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        def forward(self, x):
            a = self.backbone.norm(x.abs().permute(0, 2, 3, 4, 1))
            return {'ef': a.mean(dim=(1, 2, 3, 4)) * torch.where(x[:, 0, 0, 0, 0] < 0, -1., 1.)}
    toy = Toy().eval()
    clips = [torch.ones(1, 3, 2, 4, 4), -torch.ones(1, 3, 2, 4, 4)]
    test_rows = perturbation_core(toy, clips, fracs=(0.25,), draws=2, method='gradcam')
    assert len(test_rows) == 2 and not test_rows[0]['empty_positive'] and test_rows[1]['empty_positive']
    assert test_rows[1][0.25]['top'] is None and test_rows[1][0.25]['bottom'] is None
    assert test_rows[1]['signed_max'] < 0 and test_rows[1][0.25]['random'] >= 0
    assert not toy.backbone.norm._forward_hooks
    summary = summarise_perturbation(test_rows, fracs=(0.25,))
    assert any('1/2' in s for s in summary) and any('denominator 2' in s for s in summary)
    assert any('No ranked-map estimates' in s for s in summarise_perturbation(test_rows[1:], fracs=(0.25,)))
    print('PASS empty-positive CAM retained, no fabricated rank score, all-planned random control, mixed/all-empty denominators, cleanup')
    print('GRADCAM SELFTEST PASSED: random weights/synthetic input only, no trained checkpoint or patient data')
    return True


# ----------------------------------------------------------------------------- perturbation check
PERTURB_FRACS = (0.10, 0.25)   # my choice, not measured: share of tokens hidden
PERTURB_RANDOM_DRAWS = 5       # my choice: random masks averaged per video and fraction


def _mask_ef(model, clip, token_mask):
    """EF after setting the masked tokens' pixels to 0 in NORMALISED space (= the training mean colour)."""
    pix = token_map_to_clip(token_mask.to(torch.float32)).to(clip.device)
    with torch.no_grad():
        return float(model(clip * (1.0 - pix.view(1, 1, *pix.shape)))["ef"].float().reshape(-1))


def perturbation_core(model, clips, seed=0, fracs=PERTURB_FRACS, draws=PERTURB_RANDOM_DRAWS, method="rollout"):
    """Keep every planned clip. Empty positive CAMs have random controls but no ranked-mask score.
    Hard failures (nonfinite gradients, prediction mismatch, etc.) still stop the experiment.
    For method=both, each row contains paired rollout/gradcam records on the same clip.
    """
    if method not in ("rollout", "gradcam", "both"):
        raise ValueError("unknown attribution method")
    methods = ("rollout", "gradcam") if method == "both" else (method,)
    flows = {m: SwinAttentionFlow(model) if m == "rollout" else SwinGradCAM3D(model) for m in methods}
    out = []
    for ci, clip in enumerate(clips):
        pair = {}
        for name in methods:
            try:
                res = flows[name].run(clip)
            except EmptyPositiveCAM as e:
                # Do NOT swap targets, invert signs, select replacement clips, or rank a zero map.
                patch = tuple(model.backbone.patch_embed.proj.stride)
                shape = tuple((int(s) + int(k) - 1) // int(k) for s, k in zip(clip.shape[2:], patch))
                res = {"ef_stock": e.ef, "ef_diff": 0., "empty_positive": True,
                       "relevance": torch.zeros(shape), "signed_min": e.signed_min, "signed_max": e.signed_max}
            if res["ef_diff"] >= EF_TOL or res.get("max_block_diff", 0) >= CAPTURE_TOL:
                raise RuntimeError("capture gate failed on planned clip %d; stopping" % ci)
            rel = res["relevance"].flatten()
            n = rel.numel()
            empty = res.get("empty_positive", False)
            order = None if empty else torch.argsort(rel, descending=True, stable=True)
            gen = torch.Generator().manual_seed(seed * 100003 + ci)
            row = {"ef": res["ef_stock"], "clip_index": ci, "empty_positive": empty}
            if empty:
                row.update(signed_min=res["signed_min"], signed_max=res["signed_max"])
            for fr in fracs:
                k = int(round(fr * n))
                def mk(ids):
                    m = torch.zeros(n)
                    m[ids] = 1.0
                    return m.view(res["relevance"].shape)
                top = None if empty else abs(_mask_ef(model, clip, mk(order[:k])) - res["ef_stock"])
                bot = None if empty else abs(_mask_ef(model, clip, mk(order[-k:])) - res["ef_stock"])
                rnd = [abs(_mask_ef(model, clip, mk(torch.randperm(n, generator=gen)[:k])) - res["ef_stock"])
                       for _ in range(draws)]
                row[fr] = {"top": top, "bottom": bot, "random": sum(rnd) / len(rnd), "k": k, "n": n}
            pair[name] = row
        out.append(pair if method == "both" else pair[method])
    return out


def summarise_perturbation(rows, fracs=PERTURB_FRACS, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    lines = []
    if not rows:
        return ["No clips in this subset; no estimates or intervals."]
    all_rows = rows
    rows = [r for r in all_rows if not r.get("empty_positive", False)]
    missing = [r for r in all_rows if r.get("empty_positive", False)]
    lines.append(f"Positive-map coverage: {len(rows)}/{len(all_rows)}; empty positive CAMs: {len(missing)}/{len(all_rows)}.")
    if missing:
        lines.append("Empty-map planned positions (zero-based, not video IDs): " + str([r["clip_index"] for r in missing]))
        lines.append("Ranked-mask scores UNDEFINED for empty maps, not zero. Valid-only results below are conditional on positive-map availability; selection bias limits generalisation.")
    for fr in fracs:
        lines.append(f"All-planned random-mask mean |EF change| at {int(fr*100)}%: {np.mean([r[fr]['random'] for r in all_rows]):.3f}, denominator {len(all_rows)}")
        if not rows:
            lines.append("No ranked-map estimates: every positive map was empty.")
            continue
        top = np.array([r[fr]["top"] for r in rows])
        rnd = np.array([r[fr]["random"] for r in rows])
        bot = np.array([r[fr]["bottom"] for r in rows])
        d = top - rnd
        if len(d) > 1:
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo, hi = float("nan"), float("nan")
        lines.append(f"hide {int(fr * 100)}% of tokens ({rows[0][fr]['k']} of {rows[0][fr]['n']}), {len(rows)} clips: "
                     f"mean |EF change| most-relevant {top.mean():.3f}  random {rnd.mean():.3f}  least-relevant {bot.mean():.3f}")
        ci_text = f"[{lo:+.3f}, {hi:+.3f}]" if len(d) > 1 else "n/a (fewer than 2 usable clips)"
        lines.append(f"   most-relevant minus random: {d.mean():+.3f}  bootstrap 95% CI {ci_text}  "
                     f"most-relevant larger in {int((d > 0).sum())} of {len(d)} clips")
    return lines


def summarise_paired_perturbation(pairs, seed=0):
    """Same planned cohort, and explicitly matched usable subset. Not a full-cohort ranking."""
    lines = ["PAIRED EXPLORATORY COMPARISON: same planned clips, masks, and random controls."]
    for method in ("rollout", "gradcam"):
        lines.append("METHOD " + method + " / original planned cohort")
        lines.extend(summarise_perturbation([r[method] for r in pairs], seed=seed))
    common = [r for r in pairs if not r["gradcam"].get("empty_positive", False)]
    lines.append(f"COMMON POSITIVE-MAP SUBSET: {len(common)}/{len(pairs)}; conditional comparison, not all-clip superiority.")
    for method in ("rollout", "gradcam"):
        lines.append("METHOD " + method + " / exact same common subset")
        lines.extend(summarise_perturbation([r[method] for r in common], seed=seed))
    return lines


def run_swin_perturb(args):
    import numpy as np
    import yaml
    from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
    from src.eval.evaluate_split import (THRESHOLDS, compare_checkpoint_settings, find_checkpoint_settings,
                                         resolve_normalisation, sha256_file)
    from src.models.swin3d_multitask import SwinMultiTask

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("arch") != "swin3d_t":
        raise SystemExit(f"STOPPED: checkpoint arch is {ckpt.get('arch')!r}, expected 'swin3d_t'")
    mean, std, norm_source, _gaps = resolve_normalisation("swin3d_t", ckpt)
    _rows, mism = compare_checkpoint_settings(find_checkpoint_settings(ckpt), data_cfg)
    if mism:
        raise SystemExit("STOPPED: checkpoint and config disagree: " + "; ".join(mism))
    print(f"checkpoint sha256[:16] = {sha256_file(args.checkpoint)}  epoch = {ckpt.get('epoch')}  normalisation [{norm_source}]")
    ds = EchoNetDataset(cache_dir=data_cfg["cache_dir"], file_list_csv=data_cfg["file_list_csv"], split="VAL",
                        num_frames=data_cfg["num_frames"], frame_stride=data_cfg["frame_stride"],
                        ef_thresholds=THRESHOLDS, augment=AugmentConfig(temporal=False), seed=cfg["seed"])
    rng = np.random.default_rng(args.seed)
    idxs = sorted(int(i) for i in rng.choice(len(ds), size=args.n_videos, replace=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinMultiTask(pretrained=False, dropout_p=cfg["model"]["dropout_p"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    clips = (((ds[i]["clip"].unsqueeze(0).to(device) - mean_t) / std_t) for i in idxs)
    rows = perturbation_core(model, clips, seed=args.seed, method=args.perturb_method)
    print(f"Attribution method: {args.perturb_method}; Grad-CAM uses an interpolated coarse lattice on the shared patch grid, not new anatomical detail.")
    print(f"VAL only, {len(idxs)} clips chosen by seed {args.seed}; no TEST, no training; clip ids not printed.")
    for line in (summarise_paired_perturbation(rows, seed=args.seed) if args.perturb_method == "both" else summarise_perturbation(rows, seed=args.seed)):
        print(line)
    print("Hidden = tokens set to the normalised mean colour. Random = mean of "
          f"{PERTURB_RANDOM_DRAWS} random masks of the same size. Fractions and clip count are my choice, not measured.")
    print("Reading: larger change for most-relevant than random supports the map pointing at what the model uses; "
          "it does not show the regions are clinically meaningful. Masking also creates off-distribution input.")


# ----------------------------------------------------------------------------- matched-support temporal alignment
def temporal_alignment_score(temporal, frame_indices, keyframes, tolerance=15):
    """Compare a peak to annotations on ORIGINAL frame numbers, not clip positions.
    Random reference is exact expectation for uniform sampled-clip positions,
    including repeated positions from short-video loop padding. Ties choose first.
    No time-in-seconds or anatomical-validity claim is implied.
    """
    import numpy as np
    score = np.asarray(temporal, dtype=float)
    idx = np.asarray(frame_indices, dtype=int)
    keys = np.asarray(keyframes, dtype=int)
    if score.ndim != 1 or score.shape != idx.shape or not len(score) or not len(keys):
        raise ValueError('nonempty temporal scores/indices/annotations required')
    if not np.isfinite(score).all() or score.min() < 0 or score.max() <= 0:
        raise ValueError('finite nonnegative nonzero temporal scores required')
    distances = np.abs(idx[:, None] - keys[None, :]).min(axis=1)
    peak = int(score.argmax())
    return {'hit': float(distances[peak] <= tolerance),
            'distance': float(distances[peak]),
            'random_hit': float((distances <= tolerance).mean()),
            'peak_ties': int((score == score.max()).sum()),
            'reachable': bool((distances <= tolerance).any())}


def summarise_alignment(rows, planned, seed=0):
    import numpy as np
    lines = [f'Planned VAL clips: {planned}; annotation-eligible: {len(rows)}. No replacement sampling.']
    for method in ('rollout', 'gradcam'):
        valid = [r[method] for r in rows if r[method] is not None]
        lines.append(f'{method}: usable temporal maps {len(valid)}/{len(rows)} annotation-eligible clips; original planned denominator {planned}')
        if not valid:
            lines.append('No alignment estimate available.')
            continue
        hit = np.array([r['hit'] for r in valid]); rnd = np.array([r['random_hit'] for r in valid])
        d = hit-rnd
        rng = np.random.default_rng(seed)
        if len(valid)>1:
            boot = [d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boot,[2.5,97.5]); ci=f'[{lo:+.3f}, {hi:+.3f}]'
        else:
            ci='n/a (fewer than 2 clips)'
        lines.append(f'  peak hit {hit.mean():.3f}; matched sampled-position random expectation {rnd.mean():.3f}; lift {d.mean():+.3f}, paired bootstrap95%CI {ci}')
        lines.append(f'  mean nearest-annotation distance {np.mean([r["distance"] for r in valid]):.2f} original frames; unreachable annotations {sum(not r["reachable"] for r in valid)}/{len(valid)}; tied maxima {sum(r["peak_ties"]>1 for r in valid)}/{len(valid)}')
    common=[r for r in rows if r['gradcam'] is not None]
    lines.append(f'Common-map subset {len(common)}/{len(rows)}: conditional, not all-cohort method superiority.')
    for method in ('rollout','gradcam'):
        if common:
            lines.append(f'  {method} common-subset hit {np.mean([r[method]["hit"] for r in common]):.3f}; random {np.mean([r[method]["random_hit"] for r in common]):.3f}')
    return lines


def run_swin_alignment(args):
    import numpy as np
    import pandas as pd
    import yaml
    from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
    from src.eval.evaluate_split import (THRESHOLDS, compare_checkpoint_settings, find_checkpoint_settings,
                                         resolve_normalisation, sha256_file)
    from src.models.swin3d_multitask import SwinMultiTask
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    dc=cfg['data']; path=Path(dc['volume_tracings_csv'])
    if not path.exists():
        raise SystemExit(f'STOPPED: annotation file not found: {path}')
    annotations=pd.read_csv(path)
    if not {'FileName','Frame'}.issubset(annotations.columns):
        raise SystemExit('STOPPED: annotations require FileName and Frame columns')
    # Two distinct traced frame indices, not row count (each frame has multiple tracing rows).
    by_video={Path(str(name)).stem: np.unique(g['Frame'].to_numpy()) for name,g in annotations.groupby('FileName')}
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt.get('arch')!='swin3d_t':
        raise SystemExit('STOPPED: expected swin3d_t checkpoint')
    mean,std,source,_=resolve_normalisation('swin3d_t',ckpt)
    checks,mism=compare_checkpoint_settings(find_checkpoint_settings(ckpt),dc)
    if mism:
        raise SystemExit('STOPPED: config/checkpoint mismatch: '+str(mism))
    print(f'checkpoint sha256[:16]={sha256_file(args.checkpoint)} epoch={ckpt.get("epoch")}; normalization {source}')
    for check in checks:
        print('setting check:',check)
    ds=EchoNetDataset(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='VAL',
                      num_frames=dc['num_frames'],frame_stride=dc['frame_stride'],ef_thresholds=THRESHOLDS,
                      augment=AugmentConfig(temporal=False),seed=cfg['seed'])
    idxs=sorted(int(i) for i in np.random.default_rng(args.seed).choice(len(ds),size=args.n_videos,replace=False))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=SwinMultiTask(pretrained=False,dropout_p=cfg['model']['dropout_p'])
    m.load_state_dict(ckpt['model_state_dict']);m=m.to(device).eval()
    mean_t=torch.tensor(mean,device=device).view(1,3,1,1,1);std_t=torch.tensor(std,device=device).view(1,3,1,1,1)
    missing=invalid=empty=0; rows=[]
    for i in idxs:
        item=ds[i];stem=Path(str(item['filename'])).stem
        keys=by_video.get(stem,np.array([]))
        if len(keys)<2:
            missing+=1;continue
        total=np.load(Path(dc['cache_dir'])/(stem+'.npy'),mmap_mode='r').shape[0]
        if not np.isfinite(keys).all() or np.any(keys!=np.floor(keys)) or np.any(keys<0) or np.any(keys>=total):
            invalid+=1;continue
        clip=(item['clip'].unsqueeze(0).to(device)-mean_t)/std_t
        frame_idx=ds._sample_clip_indices(total)
        row={}
        for method in ('rollout','gradcam'):
            try:
                res=(SwinAttentionFlow(m) if method=='rollout' else SwinGradCAM3D(m)).run(clip)
            except EmptyPositiveCAM:
                row[method]=None;empty+=1;continue
            if res['ef_diff']>=EF_TOL or res.get('max_block_diff',0)>=CAPTURE_TOL:
                raise RuntimeError('attribution capture gate failed')
            pixel=res['pixel_cam'] if method=='gradcam' else token_map_to_clip(res['relevance'])
            row[method]=temporal_alignment_score(pixel.sum(dim=(1,2)).numpy(),frame_idx,keys,args.alignment_tolerance)
        rows.append(row)
    print(f'VAL only, seed{args.seed}; tolerance={args.alignment_tolerance} ORIGINAL video frames, chosen analysis parameter, not a clinical threshold.')
    print(f'No/fewer than2annotated frames: {missing}; invalid annotation frame indices: {invalid}; empty positive Grad-CAMs: {empty}.')
    for line in summarise_alignment(rows,len(idxs),args.seed):
        print(line)
    print('Exact uniform baseline uses the SAME sampled clip positions (not the whole video). Earliest argmax resolves ties; loop-padding repeats positions. No FPS assumptions.')
    print('This measures temporal proximity only, not anatomical validity or clinical correctness. Small VAL sample, exploratory; no TEST/training/new output files.')


def selftest_alignment():
    import numpy as np
    r=temporal_alignment_score([0,1,0,0],[10,12,14,16],[12,50],0)
    assert r['hit']==1 and r['random_hit']==.25 and r['distance']==0
    r=temporal_alignment_score([1,1,0],[8,10,12],[10,99],0)
    assert r['hit']==0 and r['peak_ties']==2 and r['random_hit']==1/3
    r=temporal_alignment_score([1,0,0,0],[0,2,0,2],[0,99],0)
    assert r['random_hit']==.5
    r=temporal_alignment_score([1,0],[10,12],[50,80],15)
    assert not r['reachable'] and r['random_hit']==0
    out=summarise_alignment([{'rollout':r,'gradcam':None}],2)
    assert any('0/1' in s for s in out)
    assert 'No alignment estimate available.' in out
    print('ALIGNMENT SELFTEST PASSED: original-index mapping, exact matched-support baseline, ties, loop duplicates, unreachable annotations, empty-map denominator')
    return True


def main():
    ap = argparse.ArgumentParser(description="Swin attention-flow, EF Grad-CAM, or VAL masking; synthetic selftests available")
    ap.add_argument("--selftest", action="store_true", help="synthetic checks only; no checkpoint, no data")
    ap.add_argument("--swin-rollout", action="store_true")
    ap.add_argument("--swin-perturb", action="store_true")
    ap.add_argument("--n-videos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/swin3d_t_best.pt")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--swin-gradcam", action="store_true", help="EF-gradient CAM on one VAL video")
    ap.add_argument("--selftest-gradcam", action="store_true", help="synthetic frozen-model gradient checks")
    ap.add_argument("--perturb-method", choices=("rollout", "gradcam", "both"), default="rollout")
    ap.add_argument("--swin-alignment", action="store_true", help="matched-support ED/ES proximity on VAL")
    ap.add_argument("--selftest-alignment", action="store_true")
    ap.add_argument("--alignment-tolerance", type=int, default=15)
    args = ap.parse_args()
    if sum((args.selftest, args.selftest_gradcam, args.swin_rollout, args.swin_gradcam, args.swin_perturb, args.swin_alignment, args.selftest_alignment)) > 1:
        ap.error("choose exactly one mode")
    if args.selftest_alignment:
        raise SystemExit(0 if selftest_alignment() else 1)
    if args.alignment_tolerance < 0:
        ap.error("--alignment-tolerance must be nonnegative")
    if args.swin_alignment:
        run_swin_alignment(args)
        return
    if args.selftest_gradcam:
        raise SystemExit(0 if selftest_gradcam() else 1)
    if args.n_videos < 1:
        ap.error("--n-videos must be positive")
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.swin_rollout or args.swin_gradcam:
        if not args.video_id:
            raise SystemExit("--video-id is required")
        run_swin_rollout(args)
        return
    if args.swin_perturb:
        run_swin_perturb(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
