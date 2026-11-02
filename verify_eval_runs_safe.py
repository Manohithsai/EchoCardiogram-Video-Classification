"""Read-only check of saved Echo-EF outputs. No torch, models, or pickle loading.
Run from echo-ef-project:
    python verify_eval_runs_safe.py
Optional tags follow the script name. All requested runs must exist.
This checks saved artifacts, not checkpoint provenance or inference correctness.
"""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT = ['r3d18_baseline', 'r3d18_cb_drw', 'r3d18_lds', 'swin3d_t_baseline']
ROOT = Path('results') / 'eval_runs'
EXPECTED_N = 1277
TOL = 1e-3


def sha16(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def vector(z, key):
    a = z[key]
    if a.dtype.kind not in 'fiu' or a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(f'{key} must be a finite, real numeric 1-D array')
    return a.astype(np.float64)


def main():
    tags = sys.argv[1:] or DEFAULT
    if len(tags) < 2 or len(tags) != len(set(tags)):
        raise ValueError('Supply at least two distinct run tags')
    runs = {}
    for tag in tags:
        if Path(tag).name != tag or tag in ('.', '..') or '/' in tag or '\\' in tag:
            raise ValueError(f'Expected a folder tag, not a path: {tag}')
        d = ROOT / tag
        with (d / 'metrics.json').open(encoding='utf-8') as f:
            m = json.load(f)
        pfile = d / 'predictions.npz'
        stored = m.get('predictions_sha256_16')
        if not isinstance(stored, str) or len(stored) != 16 or stored != sha16(pfile):
            raise ValueError(f'{tag}: missing or mismatched prediction-file hash')
        # Access only known fields. Never deserialize object arrays.
        with np.load(pfile, allow_pickle=False) as z:
            y, yp = vector(z, 'y_true'), vector(z, 'y_pred')
            if y.shape != yp.shape or len(y) != EXPECTED_N:
                raise ValueError(f'{tag}: expected equal 1-D arrays of {EXPECTED_N} cases')
            names = None
            if 'file_names' in z.files:
                try:
                    a = z['file_names']
                except ValueError:
                    print(f'{tag}: file_names is not safely loadable; ID alignment unverified')
                else:
                    if a.ndim != 1 or len(a) != len(y) or a.dtype.kind not in 'US':
                        raise ValueError(f'{tag}: file_names must be a 1-D string array matching y_true')
                    names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in a.tolist()]
                    if any(not x.strip() for x in names) or len(set(names)) != len(names):
                        raise ValueError(f'{tag}: empty or duplicate video IDs')
        err = yp - y
        sst = np.sum((y - y.mean()) ** 2)
        if sst <= 0:
            raise ValueError(f'{tag}: undefined R2 (zero target variance)')
        actual = {
            'mae': float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'r2_coefficient_of_determination': float(1 - np.sum(err ** 2) / sst),
        }
        for key, value in actual.items():
            saved = m.get(key)
            if isinstance(saved, bool) or not isinstance(saved, (float, int)) or not np.isfinite(saved):
                raise ValueError(f'{tag}: missing/non-finite metric {key}')
            if abs(value - saved) > TOL:
                raise ValueError(f'{tag}: {key} mismatch: recomputed={value}, saved={saved}')
        print(f'{tag}: hash/metrics OK; n={len(y)}; MAE={actual["mae"]:.6f}; '
              f'RMSE={actual["rmse"]:.6f}; R2={actual["r2_coefficient_of_determination"]:.6f}; '
              f'target_variance={sst / len(y):.6f}')
        runs[tag] = {'y': y, 'yp': yp, 'names': names, 'm': m}
    print('\nCheckpoint hashes below are stored claims only; checkpoint files are NOT verified.')
    for a, b in combinations(runs, 2):
        ra, rb = runs[a], runs[b]
        ha, hb = ra['m'].get('checkpoint_sha256_16'), rb['m'].get('checkpoint_sha256_16')
        print(f'{a} / {b}: stored checkpoint hashes {ha!r} / {hb!r}')
    # Every requested run must be aligned before any cross-run prediction comparison.
    ref_tag = tags[0]
    ref = runs[ref_tag]
    ids_verified = all(r['names'] is not None for r in runs.values())
    for tag, r in runs.items():
        if not np.array_equal(r['y'], ref['y']):
            raise ValueError(f'{tag}: target labels/order differ from {ref_tag}; comparison stopped')
        if ids_verified and r['names'] != ref['names']:
            raise ValueError(f'{tag}: video IDs/order differ from {ref_tag}; comparison stopped')
    if ids_verified:
        print('\nAll requested runs: identical saved IDs/order and target labels.')
    else:
        print('\nWARNING: video IDs were not safely loadable (saved as an object array), so ID alignment is '
              'NOT verified. All 1277 target labels are identical across runs, which supports but does not '
              'prove the same video order. The test loader uses shuffle=False; confirm the order against '
              'the test file list separately.')
    for a, b in combinations(runs, 2):
        pa, pb = runs[a]['yp'], runs[b]['yp']
        print(f'{a} / {b}: identical prediction VALUES={np.array_equal(pa, pb)}; '
              f'max_abs_difference={np.max(np.abs(pa - pb)):.9g}')
    a, b = 'r3d18_baseline', 'swin3d_t_baseline'
    if a in runs and b in runs:
        y = runs[a]['y']
        delta = np.abs(runs[b]['yp'] - y) - np.abs(runs[a]['yp'] - y)
        rng = np.random.default_rng(0)
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print('\nExploratory paired percentile bootstrap: MAE(Swin) - MAE(R3D)')
        print(f'difference={delta.mean():+.6f}; 95% interval=[{lo:+.6f}, {hi:+.6f}]')
        print('Negative favors Swin. An interval containing zero is NOT evidence of equivalence.')
        print('Video-level resampling assumes independent videos; cluster by patient if needed.')
    else:
        print('R3D/Swin pair not requested; bootstrap not performed.')
    print('\nSaved-output consistency checks passed. This does NOT prove correct inference, '
          'an untouched test set, checkpoint identity, normalization, or clinical validity.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f'\nSTOPPED: {e}', file=sys.stderr)
        sys.exit(1)