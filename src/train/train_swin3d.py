"""
Train Video Swin (Swin3D-T) for EF / EDV / ESV with the SAME recipe as the R3D-18 baseline:
unweighted smooth-L1, EF-consistency loss from the config, frozen-backbone warm-up then
unfreeze, checkpoint = lowest OVERALL validation MAE, early stopping on validation.
(So the R3D-18 vs Video Swin comparison differs only in the architecture.)

Run from echo-ef-project/ :

  1) memory / speed check (3 batches, then exits):
     python -m src.train.train_swin3d --config configs/base.yaml --dry-run --batch-size 4

  2) full training (use the batch size that fit in step 1; --grad-accum keeps the
     effective batch size = batch-size * grad-accum):
     python -m src.train.train_swin3d --config configs/base.yaml --batch-size 4 --grad-accum 2

Bounded MixUp experiment mode (added 2026-11-07; everything above is the original usage,
unchanged):
  One arm per launch, full precision, on the four-class ablation's exact 643/645
  selection/calibration split (read from the sealed v4 protocol; never recomputed here):

  python -m src.train.train_swin3d --config configs/base.yaml --control --fp32 --acknowledge-protocol
  python -m src.train.train_swin3d --config configs/base.yaml --mixup   --fp32 --acknowledge-protocol

  20-epoch cap, 4 backbone-frozen epochs, patience 6, batch 4, seed 42, centered clips,
  selection = lowest unrounded selection-half EF MAE (strict improvement; earliest epoch
  wins ties). Both arms share one saved initialization (results/swin_mixup_v1/initial.pt),
  created once from Kinetics-400 pretrained weights. No TEST or calibration-half evaluation.
"""
import argparse
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.echonet_dataset import AugmentConfig, EchoNetDataset
from src.models.r3d_multitask import ef_consistency_loss
from src.models.swin3d_multitask import SwinMultiTask, swin_norm_stats

CKPT_NAME = "swin3d_t_best.pt"
RUN_NAME = "swin3d-t"

# Bounded MixUp experiment constants (mirrors the R3D four-class ablation protocol).
MIXUP_CUTOFFS = (40.0, 50.0, 70.0)  # four-class boundaries, used ONLY to reproduce the ablation split
MIXUP_EPOCH_CAP = 20
MIXUP_FROZEN_EPOCHS = 4
MIXUP_PATIENCE = 6
MIXUP_SEED = 42
# Bounded path loads single-process: AblationDataset is a local class, which Windows spawn
# workers cannot pickle; workers=0 also matches the R3D four-class ablation's loader setting.
BOUNDED_NUM_WORKERS = 0


def normalize(clip, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (clip - mean_t) / std_t


def to_class(ef_np, thresholds):
    cls = np.zeros(len(ef_np), dtype=int)
    for t in thresholds:
        cls += (ef_np >= t).astype(int)
    return cls


def centered_indices(total_frames, num_frames=32, stride=2):
    """Deterministic centred window; loop-pad short videos. Matches the dataset's eval path
    and the four-class ablation's clip framing (no random start when temporal jitter is off)."""
    span = num_frames * stride
    if total_frames <= span:
        return (np.arange(num_frames) * stride) % total_frames
    start = (total_frames - span) // 2
    return start + np.arange(num_frames) * stride


def mixup_batch_swin(clip, batch, permutation, retained):
    """Same semantics as the R3D four-class ablation's mixup arm: interpolate the raw
    [0,1] clips and ALL regression targets with one Beta(0.4,0.4) lambda per batch.
    Swin has no class head, so there are no soft class targets to mix."""
    if not 0 <= retained <= 1:
        raise ValueError('retained fraction must be in [0,1]')
    mixed_clip = retained * clip + (1 - retained) * clip[permutation]
    mixed = {k: retained * batch[k] + (1 - retained) * batch[k][permutation]
             for k in ('ef', 'edv', 'esv')}
    return mixed_clip, mixed


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(value, path):
    p = Path(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, p)


def atomic_checkpoint(value, path):
    p = Path(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, p)


def deterministic_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ablation_split(path):
    """Read the sealed four-class v4 protocol's selection/calibration positions so this
    experiment validates on the EXACT same 643 videos. No recomputation, no new split."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f'STOPPED: ablation protocol not found at {p}; run from the project folder')
    proto = json.loads(p.read_text())
    sel, cal = proto.get('selection_positions'), proto.get('calibration_positions')
    if not isinstance(sel, list) or not isinstance(cal, list):
        raise SystemExit('STOPPED: ablation protocol lacks selection/calibration positions')
    if len(sel) != 643 or len(cal) != 645:
        raise SystemExit(f'STOPPED: unexpected split sizes {len(sel)}/{len(cal)}; expected 643/645')
    if set(sel) & set(cal):
        raise SystemExit('STOPPED: ablation split positions overlap; do not improvise a split')
    return sel, cal


def experiment_protocol(cfg, sel, cal, fp32, mean, std, split_source):
    return {
        'version': 'swin-mixup-fp32-v1',
        'seed': MIXUP_SEED,
        'arms': ['control', 'mixup'],
        'epoch_cap': MIXUP_EPOCH_CAP,
        'frozen_epochs': MIXUP_FROZEN_EPOCHS,
        'patience': MIXUP_PATIENCE,
        'batch_size': 4,
        'workers': BOUNDED_NUM_WORKERS,
        'pretrained': 'Swin3D_T_Weights.KINETICS400_V1',
        'selection_metric': 'unrounded selection-half EF MAE; strict improvement; earliest epoch wins ties',
        'loss': 'EF/EDV/ESV smooth-L1 + config consistency weight; mixup arm interpolates clips and all three targets (Beta 0.4,0.4 per batch)',
        'sampler': 'plain shuffle, per-epoch seeded generator; no class reweighting (Swin has no class head)',
        'augmentations': {'control': 'centered 32-frame clip stride 2; no augmentation',
                          'mixup': 'control + batch mixup lam~Beta(0.4,0.4) on raw clips/targets'},
        'norm_mean': list(mean), 'norm_std': list(std),
        'numerics': ('FP32 forward/loss/backward; autocast/scaling/TF32 disabled; finite loss/gradient '
                     'checks; hard stop on failure' if fp32 else
                     'AMP autocast + GradScaler (config train.amp); hard stop on nonfinite loss/gradient'),
        'config': cfg,
        'selection_positions': sel,
        'calibration_positions': cal,
        'split_source': str(split_source),
        'file_list_sha256': file_hash(cfg['data']['file_list_csv']),
        'source_sha256': file_hash(__file__),
        'model_source_sha256': file_hash(Path(__file__).parents[1] / 'models' / 'swin3d_multitask.py'),
        'dataset_source_sha256': file_hash(Path(__file__).parents[1] / 'data' / 'echonet_dataset.py'),
        # Plain str: torch.__version__ is a TorchVersion str subclass on some torch builds;
        # embedded raw it pickles as a GLOBAL that weights_only loads reject (torch 2.6-2.11).
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__)},
        'caveats': ['one seed, matched-pair experiment (control vs mixup); not a repeat of the five-arm ablation',
                    'validation videos are the ablation selection half; VAL was used earlier in the project',
                    'calibration half is reserved, not evaluated here; TEST never evaluated here',
                    'Swin recipe keeps plain shuffle and its own normalisation; differs from the R3D ablation sampler/head by design'],
    }


def run_epoch(model, loader, optimizer, scaler, cfg, device, mean, std, train, accum=1, max_batches=None,
              timings=None, amp_enabled=True, mixup_rng=None):
    model.train() if train else model.eval()
    cw = cfg["model"]["ef_consistency_weight"]
    thresholds = cfg["data"]["ef_thresholds"]
    total_loss, n_batches = 0.0, 0
    preds, trues = [], []
    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            clip = batch["clip"].to(device)
            targets = {k: batch[k].to(device) for k in ("ef", "edv", "esv")}
            if train and mixup_rng is not None:
                # Mix on raw [0,1] clips before normalisation, exactly like the R3D ablation.
                perm = torch.tensor(mixup_rng.permutation(len(clip)), device=device)
                lam = float(mixup_rng.beta(0.4, 0.4))
                clip, targets = mixup_batch_swin(clip, targets, perm, lam)
            clip = normalize(clip, mean, std, device)
            ef, edv, esv = targets["ef"], targets["edv"], targets["esv"]

            with torch.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                out = model(clip)
                loss = (nn.functional.smooth_l1_loss(out["ef"], ef)
                        + nn.functional.smooth_l1_loss(out["edv"], edv)
                        + nn.functional.smooth_l1_loss(out["esv"], esv))
                if cw > 0:
                    loss = loss + cw * ef_consistency_loss(out["ef"], out["edv"], out["esv"])

            if train:
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss; hard stop, no silent skip')
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                    if not torch.isfinite(norm):
                        raise RuntimeError('nonfinite gradients; hard stop, no silent skip')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            lv = loss.item()
            if timings is not None:  # loss.item() already synchronised the GPU
                timings.append(time.time())
            if np.isfinite(lv):
                total_loss += lv
                n_batches += 1
                preds.append(out["ef"].detach().float().cpu().numpy())
                trues.append(batch["ef"].numpy())

    pred, true = np.concatenate(preds), np.concatenate(trues)
    err = np.abs(pred - true)
    cls = to_class(true, thresholds)
    per_class = [float(err[cls == c].mean()) if (cls == c).any() else float("nan") for c in range(len(thresholds) + 1)]
    return {"loss": total_loss / max(n_batches, 1), "mae": float(err.mean()),
            "per_class_mae": per_class, "class_avg_mae": float(np.nanmean(per_class))}


def _allowlist_torch_version():
    # Checkpoints written by the first version of this experiment embedded the raw
    # torch.__version__ (a TorchVersion str subclass), which torch 2.6-2.11 do not
    # allowlist for weights_only loads. TorchVersion is torch's own str subclass and
    # benign; this lets those already-written checkpoints load. New checkpoints store
    # a plain str instead, so this is belt-and-braces.
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass


def bounded_experiment(args):
    """One arm per launch, fp32, ablation split, shared saved initialization. Mirrors the
    R3D four-class ablation's safeguards: protocol guard, run lock, atomic writes,
    epoch-boundary resume, fail-closed on any nonfinite value."""
    if not args.acknowledge_protocol:
        raise SystemExit('STOPPED: read protocol; --acknowledge-protocol required for bounded training')
    if not torch.cuda.is_available():
        raise SystemExit('STOPPED: CUDA required, no silent CPU run')
    _allowlist_torch_version()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg['data']['num_frames'] != 32 or cfg['data']['frame_stride'] != 2 or cfg['data']['frame_size'] != 112:
        raise SystemExit('STOPPED: bounded protocol requires 32 frames / stride 2 / 112px; do not silently change recipe')
    amp_enabled = bool(cfg['train']['amp']) and not args.fp32
    deterministic_seed(MIXUP_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')

    class AblationDataset(EchoNetDataset):
        def _sample_clip_indices(self, total_frames):
            if self.augment.temporal:
                return super()._sample_clip_indices(total_frames)
            return centered_indices(total_frames, self.num_frames, self.frame_stride)

    dc = cfg['data']
    train_ds = AblationDataset(cache_dir=dc['cache_dir'], file_list_csv=dc['file_list_csv'], split='TRAIN',
                               num_frames=32, frame_stride=2, ef_thresholds=dc['ef_thresholds'],
                               augment=AugmentConfig(temporal=False), seed=MIXUP_SEED)
    val_ds = EchoNetDataset(cache_dir=dc['cache_dir'], file_list_csv=dc['file_list_csv'], split='VAL',
                            num_frames=32, frame_stride=2, ef_thresholds=dc['ef_thresholds'],
                            augment=AugmentConfig(temporal=False), seed=MIXUP_SEED)
    sel, cal = load_ablation_split(args.split_from)
    if max(sel + cal) >= len(val_ds):
        raise SystemExit('STOPPED: ablation split positions exceed VAL rows; FileList mismatch')

    mean, std = swin_norm_stats()
    protocol = experiment_protocol(cfg, sel, cal, args.fp32, mean, std, args.split_from)
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit('STOPPED: protocol/source/data/runtime differs; do not reuse this run namespace')
        if not args.resume:
            raise SystemExit('STOPPED: run exists; use --resume for same protocol')
    else:
        if any(root.iterdir()):
            raise SystemExit('STOPPED: output directory not empty without a protocol')
        if args.resume:
            raise SystemExit('STOPPED: no existing protocol to resume')
        atomic_json(protocol, protocol_path)

    lock = root / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('STOPPED: RUNNING.lock exists; verify old process stopped before removing stale lock')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    arm = 'mixup' if args.mixup else 'control'
    try:
        initial = root / 'initial.pt'
        if not initial.exists():
            # One shared starting point for both arms: seeded Kinetics-400 pretrained init.
            deterministic_seed(MIXUP_SEED)
            m0 = SwinMultiTask(pretrained=True, dropout_p=cfg['model']['dropout_p'])
            atomic_checkpoint({'model_state_dict': m0.state_dict(), 'protocol': protocol}, initial)
            del m0
            print('Created shared Swin initialization from Kinetics-400 pretrained weights.', flush=True)
        init = torch.load(initial, map_location='cpu', weights_only=True)
        if init['protocol'] != protocol:
            raise RuntimeError('initial checkpoint protocol mismatch')
        init_hash = file_hash(initial)

        print(f"Staged Swin arm '{arm}';20epochcap4frozen,patience6;fp32; no automatic other arm.", flush=True)
        print(f'TRAIN size {len(train_ds)}; selection/calibration sizes: {len(sel)} {len(cal)}; no TEST/calibration evaluation.', flush=True)

        val_loader = DataLoader(Subset(val_ds, sel), batch_size=4, shuffle=False,
                                num_workers=BOUNDED_NUM_WORKERS, pin_memory=True)

        active = root / arm
        active.mkdir(exist_ok=True)
        if (active / 'complete.json').exists():
            complete = json.loads((active / 'complete.json').read_text())
            if file_hash(active / 'best.pt') != complete['best_sha256']:
                raise RuntimeError('completed checkpoint hash mismatch')
            print('Already complete, not rerunning:', arm, flush=True)
            return

        model = SwinMultiTask(pretrained=False, dropout_p=cfg['model']['dropout_p']).to(device)
        model.load_state_dict(init['model_state_dict'])
        optimizer = torch.optim.AdamW(
            [{'params': model.head.parameters(), 'lr': cfg['train']['lr_head']},
             {'params': model.backbone.parameters(), 'lr': cfg['train']['lr_backbone']}],
            weight_decay=cfg['train']['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled, init_scale=2.0 ** 12)

        start, best, patience, history = 0, float('inf'), 0, []
        if (active / 'last.pt').exists() or (active / 'best.pt').exists():
            candidates = [torch.load(active / name, map_location='cpu', weights_only=True)
                          for name in ('last.pt', 'best.pt') if (active / name).exists()]
            state = max(candidates, key=lambda s: s['epoch'])
            if state['initial_sha256'] != init_hash or state['protocol'] != protocol:
                raise RuntimeError('resume provenance mismatch')
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            start, best, patience, history = state['epoch'], state['best_mae'], state['patience'], state['history']
            del candidates, state

        for epoch in range(start, MIXUP_EPOCH_CAP):
            if patience >= MIXUP_PATIENCE:
                break
            # Per-epoch seed supports exact epoch-boundary resume; an interrupted epoch reruns.
            deterministic_seed(MIXUP_SEED + epoch)
            mixup_rng = np.random.default_rng(4242 + epoch) if args.mixup else None
            generator = torch.Generator().manual_seed(MIXUP_SEED + epoch)
            loader = DataLoader(train_ds, batch_size=4, shuffle=True, generator=generator,
                                num_workers=BOUNDED_NUM_WORKERS, pin_memory=True, drop_last=True)
            frozen = epoch < MIXUP_FROZEN_EPOCHS
            for p in model.backbone.parameters():
                p.requires_grad_(not frozen)
            if frozen:
                model.backbone.eval()

            t0 = time.perf_counter()
            tr = run_epoch(model, loader, optimizer, scaler, cfg, device, mean, std, True,
                           args.grad_accum, amp_enabled=amp_enabled, mixup_rng=mixup_rng)

            model.eval()
            va = run_epoch(model, val_loader, optimizer, scaler, cfg, device, mean, std, False,
                           amp_enabled=False)  # validation fixed fp32 across arms, as in the ablation
            met = {'ef_mae': va['mae'], 'per_class_mae': va['per_class_mae'],
                   'class_avg_mae': va['class_avg_mae']}
            row = {'epoch': epoch + 1, 'train_loss': tr['loss'], 'selection': met,
                   'seconds': time.perf_counter() - t0}
            history.append(row)
            improved = met['ef_mae'] < best
            if improved:
                best, patience = met['ef_mae'], 0
            else:
                patience += 1
            payload = {'arch': 'swin3d_t_mixup_experiment', 'arm': arm,
                       'model_state_dict': model.state_dict(), 'epoch': epoch + 1,
                       'val_mae': met['ef_mae'], 'norm_mean': list(mean), 'norm_std': list(std),
                       'protocol': protocol, 'initial_sha256': init_hash,
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scaler_state_dict': scaler.state_dict(),
                       'best_mae': best, 'patience': patience, 'history': history}
            if improved:
                atomic_checkpoint(payload, active / 'best.pt')
            atomic_checkpoint(payload, active / 'last.pt')
            atomic_json(history, active / 'history.json')
            print(f'{arm} epoch{epoch + 1}/20 selectionMAE {met["ef_mae"]:.4f} '
                  f'classAvgMAE {met["class_avg_mae"]:.4f} best {best:.4f} '
                  f'patience{patience}/6 minutes{row["seconds"] / 60:.2f}', flush=True)

        best_state = torch.load(active / 'best.pt', map_location='cpu', weights_only=True)
        atomic_json({'arm': arm, 'best_epoch': best_state['epoch'], 'best_selection_mae': best,
                     'selection_metrics': best_state['history'][best_state['epoch'] - 1]['selection'],
                     'best_sha256': file_hash(active / 'best.pt'), 'epochs_completed': len(history)},
                    active / 'complete.json')
        print(f"ARM COMPLETE: ['{arm}'] best epoch{best_state['epoch']} selectionMAE {best:.4f}. "
              'Review before any other run. No automatic next run.', flush=True)
    except BaseException as e:
        atomic_json({'status': 'interrupted' if isinstance(e, KeyboardInterrupt) else 'failed',
                     'arm': arm, 'error': str(e), 'traceback': traceback.format_exc(),
                     'resume': 'epoch-boundary only; interrupted epoch reruns'},
                    root / 'last_failure.json')
        raise
    finally:
        lock.unlink(missing_ok=True)


def selftest():
    """CPU-only synthetic checks: mixup math, centred framing, split loading guard, atomic
    roundtrip, Swin forward/loss/gradient shapes. No data, no downloads, no training."""
    torch.manual_seed(MIXUP_SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    _allowlist_torch_version()
    x = torch.stack((torch.zeros(3, 4, 4, 4), torch.ones(3, 4, 4, 4)))
    batch = {'ef': torch.tensor([30., 80.]), 'edv': torch.tensor([100., 120.]), 'esv': torch.tensor([70., 24.])}
    perm = torch.tensor([1, 0])
    xm, tm = mixup_batch_swin(x, batch, perm, 0.25)
    assert torch.all(xm[0] == 0.75) and torch.all(xm[1] == 0.25)
    assert tm['ef'][0] == 0.25 * 30 + 0.75 * 80 and tm['edv'][1] == 0.25 * 120 + 0.75 * 100
    assert centered_indices(10).tolist() == ((np.arange(32) * 2) % 10).tolist()
    assert centered_indices(100)[0] == (100 - 64) // 2 and len(centered_indices(100)) == 32
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'state.pt'
        atomic_checkpoint({'state': torch.tensor([1., 2.]), 'epoch': 3}, p)
        assert torch.equal(torch.load(p, weights_only=True)['state'], torch.tensor([1., 2.]))
        # rev2: new checkpoints carry a plain-str torch version...
        atomic_checkpoint({'protocol': {'runtime': {'torch': str(torch.__version__)}},
                           'state': torch.tensor([1.])}, p)
        assert torch.load(p, weights_only=True)['protocol']['runtime']['torch'] == str(torch.__version__)
        # ...and a v1-style checkpoint with the raw TorchVersion global still loads,
        # which is exactly the recovery path for initial.pt files written before rev2.
        atomic_checkpoint({'protocol': {'runtime': {'torch': torch.__version__}},
                           'state': torch.tensor([2.])}, p)
        legacy = torch.load(p, weights_only=True)
        assert str(legacy['protocol']['runtime']['torch']) == str(torch.__version__)
        assert legacy['protocol']['runtime']['torch'] == str(torch.__version__)
        atomic_json({'complete': False}, Path(d) / 'test.json')
        proto = {'selection_positions': list(range(643)), 'calibration_positions': list(range(643, 1288))}
        (Path(d) / 'proto.json').write_text(json.dumps(proto))
        sel, cal = load_ablation_split(Path(d) / 'proto.json')
        assert len(sel) == 643 and len(cal) == 645
    # Windows-spawn regression guard: a local dataset class cannot be pickled into spawn
    # workers (this exact failure crashed the laptop run); workers=0 must iterate fine and
    # must stay the bounded setting.
    assert BOUNDED_NUM_WORKERS == 0
    import pickle
    class LocalDS(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return torch.tensor([float(i)])
    assert next(iter(DataLoader(LocalDS(), batch_size=1, num_workers=0))) is not None
    try:
        next(iter(DataLoader(LocalDS(), batch_size=1, num_workers=1,
                             multiprocessing_context='spawn')))
        raise AssertionError('spawn worker unexpectedly pickled a local class')
    except (AttributeError, pickle.PicklingError, EOFError, RuntimeError):
        pass
    model = SwinMultiTask(pretrained=False, dropout_p=0.3).eval()
    out = model(torch.randn(1, 3, 8, 32, 32))
    assert out['ef'].shape == (1,) and torch.isfinite(out['ef']).all()
    model.train()
    out = model(torch.randn(1, 3, 8, 32, 32))
    loss = F.smooth_l1_loss(out['ef'], torch.tensor([55.])) + F.smooth_l1_loss(out['edv'], torch.tensor([120.])) \
        + F.smooth_l1_loss(out['esv'], torch.tensor([54.]))
    loss.backward()
    assert torch.isfinite(loss) and model.head[-1].weight.grad is not None
    print('SELFTEST PASSED: mixup math, centred framing, split guard, atomic roundtrip, Swin forward/backward. '
          'Synthetic only; no data, no training.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=None, help="override cfg batch size (6GB GPU: try 4 or 2)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-backbone", type=float, default=None, help="override backbone LR (Swin usually wants <= 1e-4)")
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="run a short benchmark, print GPU memory/time, exit")
    ap.add_argument("--bench-batches", type=int, default=24, help="batches to time in --dry-run (first 4 are ignored)")
    # Bounded MixUp experiment mode (one arm per launch).
    ap.add_argument("--mixup", action="store_true", help="bounded experiment: mixup arm")
    ap.add_argument("--control", action="store_true", help="bounded experiment: matched no-mixup control arm")
    ap.add_argument("--fp32", action="store_true", help="bounded experiment: full precision (disables config AMP)")
    ap.add_argument("--acknowledge-protocol", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-dir", default='results/swin_mixup_v1')
    ap.add_argument("--split-from", default='results/fourclass_fp32_v4/protocol.json',
                    help="sealed ablation protocol.json supplying the exact 643/645 split")
    ap.add_argument("--selftest", action="store_true", help="CPU synthetic checks, then exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.mixup or args.control:
        if args.mixup and args.control:
            raise SystemExit('STOPPED: choose exactly one arm: --mixup or --control')
        if not args.fp32:
            raise SystemExit('STOPPED: bounded experiment is fp32; pass --fp32 explicitly')
        bounded_experiment(args)
        return

    cfg = yaml.safe_load(open(args.config))
    bs = args.batch_size or cfg["train"]["batch_size"]
    lr_bb = args.lr_backbone or cfg["train"]["lr_backbone"]
    lr_head = args.lr_head or cfg["train"]["lr_head"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = swin_norm_stats()
    print(f"device={device} batch_size={bs} grad_accum={args.grad_accum} lr_backbone={lr_bb} lr_head={lr_head}")
    print(f"normalisation mean={mean} std={std}")

    def make_ds(split, aug):
        return EchoNetDataset(
            cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"], split=split,
            num_frames=cfg["data"]["num_frames"], frame_stride=cfg["data"]["frame_stride"],
            ef_thresholds=cfg["data"]["ef_thresholds"], augment=AugmentConfig(temporal=aug), seed=cfg["seed"])

    train_ds, val_ds = make_ds("TRAIN", True), make_ds("VAL", False)
    print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg["train"]["num_workers"],
                            pin_memory=True)

    model = SwinMultiTask(pretrained=True, dropout_p=cfg["model"]["dropout_p"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SwinMultiTask parameters: {n_params:.1f}M")

    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": lr_head},
         {"params": model.backbone.parameters(), "lr": lr_bb}],
        weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda", init_scale=2.0 ** 12)

    if args.dry_run:
        for p in model.backbone.parameters():
            p.requires_grad = True  # worst case for memory
        stamps = [time.time()]
        run_epoch(model, train_loader, optimizer, scaler, cfg, device, mean, std, True, args.grad_accum,
                  max_batches=args.bench_batches, timings=stamps)
        gaps = np.diff(stamps)  # seconds per batch, includes data loading
        steady = gaps[4:] if len(gaps) > 6 else gaps  # skip start-up batches (CUDA init, worker spawn)
        med, mean_gap = float(np.median(steady)), float(np.mean(steady))
        if device.type == "cuda":
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (of ~6 GB)")
        print(f"first 4 batches (start-up): {', '.join(f'{g:.1f}s' for g in gaps[:4])}")
        print(f"steady state over {len(steady)} batches: median {med:.2f} s, mean {mean_gap:.2f} s per batch of {bs}")
        n_b = len(train_loader)
        print(f"-> ~{med * n_b / 60:.0f} min per epoch (train only, median) / ~{mean_gap * n_b / 60:.0f} min (mean); "
              f"validation adds roughly 15-20% more")
        print("Dry run OK. If memory is close to 6 GB, lower --batch-size and raise --grad-accum.")
        return

    import wandb
    wandb.init(project="echo-ef", name=RUN_NAME,
               config={**cfg, "arch": "swin3d_t", "batch_size": bs, "grad_accum": args.grad_accum,
                       "lr_backbone": lr_bb, "lr_head": lr_head})
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, patience = float("inf"), 0
    warmup = cfg["train"]["warmup_epochs"]

    for epoch in range(cfg["train"]["epochs"]):
        if epoch == warmup:
            print(f"[epoch {epoch + 1}] Unfreezing backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True
        tr = run_epoch(model, train_loader, optimizer, scaler, cfg, device, mean, std, True, args.grad_accum)
        va = run_epoch(model, val_loader, optimizer, scaler, cfg, device, mean, std, False)
        pc = va["per_class_mae"]
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']} | train_mae={tr['mae']:.3f} | val_mae={va['mae']:.3f} | "
              f"val_class_avg_mae={va['class_avg_mae']:.3f} | per-class R/M/N = " + "/".join(f"{x:.2f}" for x in pc))
        log = {"epoch": epoch + 1, "train_loss": tr["loss"], "train_mae": tr["mae"], "val_loss": va["loss"],
               "val_mae": va["mae"], "val_class_avg_mae": va["class_avg_mae"],
               "backbone_frozen": epoch < warmup}
        for i, x in enumerate(pc):
            log[f"val_mae_class{i}"] = x
        wandb.log(log)

        if va["mae"] < best:  # same selection rule as the R3D-18 baseline: overall val MAE
            best, patience = va["mae"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_mae": va["mae"],
                        "val_class_avg_mae": va["class_avg_mae"], "arch": "swin3d_t", "config": cfg,
                        "norm_mean": mean, "norm_std": std}, ckpt_dir / CKPT_NAME)
            print(f"  -> New best val_mae={best:.3f}. Saved to {ckpt_dir / CKPT_NAME}")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    wandb.finish()
    print(f"\nDone. Best overall val MAE: {best:.3f}")


if __name__ == "__main__":
    main()
