"""Read-only corroboration of saved TEST labels against the current FileList.csv.
Save as check_test_manifest.py in echo-ef-project, then run:
    python check_test_manifest.py
No model, checkpoint, cache, pickle loading, writes, or bootstrap.
Paths below come from the supplied configs/base.yaml. This does not parse YAML.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CSV_PATH = Path('data/EchoNet-Dynamic/FileList.csv')
ROOT = Path('results/eval_runs')
DEFAULT = ['r3d18_baseline', 'r3d18_cb_drw', 'r3d18_lds', 'swin3d_t_baseline']
KEYS = [('EF', 'y_true'), ('EDV', 'edv_true'), ('ESV', 'esv_true')]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    tags = sys.argv[1:] or DEFAULT
    if not tags or len(tags) != len(set(tags)):
        raise ValueError('Run tags must be distinct')
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'FileName', 'Split', 'EF', 'EDV', 'ESV'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV missing required columns')
        rows = list(reader)
    ids = [r['FileName'] for r in rows]
    if any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Empty or repeated FileName in CSV; cannot establish unique video rows')
    # Dataset cache keys use Path(FileName).stem. Check collisions as well.
    stems = [Path(x).stem for x in ids]
    if len(set(stems)) != len(stems):
        raise ValueError('CSV names collide at cache-file stem level')
    splits = [r['Split'].upper() for r in rows]
    if any(s not in {'TRAIN', 'VAL', 'TEST'} for s in splits):
        raise ValueError('Unexpected Split value; inspect current CSV')
    split_ids = {s: {r['FileName'] for r in rows if r['Split'].upper() == s}
                 for s in ('TRAIN', 'VAL', 'TEST')}
    if any(split_ids[a] & split_ids[b] for a, b in
           [('TRAIN', 'VAL'), ('TRAIN', 'TEST'), ('VAL', 'TEST')]):
        raise ValueError('Video IDs overlap across splits')
    print(f'Current CSV SHA256: {sha256(CSV_PATH)}')
    print('CSV split counts:', dict(Counter(splits)))
    print('Video IDs and cache stems unique; video-level splits disjoint. Patient independence not checked.')
    test = [r for r in rows if r['Split'].upper() == 'TEST']
    if len(test) != 1277:
        raise ValueError(f'Expected 1277 TEST rows; found {len(test)}')
    # EchoNetDataset converts each label to torch.float32 before saving.
    expected = np.array([[float(r[col]) for col, _ in KEYS] for r in test],
                        dtype=np.float32).astype(np.float64)
    if not np.isfinite(expected).all():
        raise ValueError('Non-finite TEST labels')
    triples = [tuple(x) for x in expected]
    counts = Counter(triples)
    ambiguous_rows = sum(n for n in counts.values() if n > 1)
    print(f'TEST target triples: {len(counts)} unique values across 1277 rows; '
          f'{ambiguous_rows} rows belong to repeated triples.')
    for tag in tags:
        if Path(tag).name != tag or tag in ('.', '..') or '/' in tag or '\\' in tag:
            raise ValueError(f'Expected folder tag, not a path: {tag}')
        d = ROOT / tag
        pfile = d / 'predictions.npz'
        with (d / 'metrics.json').open(encoding='utf-8') as f:
            m = json.load(f)
        current_hash = sha256(pfile)
        if m.get('predictions_sha256_16') != current_hash[:16]:
            raise ValueError(f'{tag}: stored prediction-file checksum missing/mismatched')
        with np.load(pfile, allow_pickle=False) as z:
            for column, key in KEYS:
                a = z[key]
                if a.ndim != 1 or len(a) != len(test) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
                    raise ValueError(f'{tag}: invalid numeric vector {key}')
                want = expected[:, [k for _, k in KEYS].index(key)]
                same = a.astype(np.float64) == want
                if not same.all():
                    i = int(np.flatnonzero(~same)[0])
                    raise ValueError(f'{tag}: {column} mismatch at row {i}; saved={a[i]!r}, '
                                     f'current_CSV_float32={want[i]!r}. No cross-run pairing established.')
        print(f'{tag}: checksum OK; EF/EDV/ESV all match current TEST rows EXACTLY and in order.')
    print('\nCURRENT-MANIFEST CORROBORATION COMPLETE for all requested runs.')
    if ambiguous_rows:
        print('AMBIGUOUS IDENTITY: repeated target triples cannot distinguish those video rows. '
              'Do not call all historical video IDs verified or use this alone to validate paired inference.')
    else:
        print('Every TEST target triple uniquely identifies a row in the CURRENT CSV. '
              'This strongly corroborates common order under the inspected dataset/evaluator code.')
    print('This reconstructed evidence is not extraction of the old saved IDs. It cannot prove the '
          'historical CSV/code was unchanged, checkpoint identity, correct inference, or an untouched test.')
    print('No paired bootstrap was performed. Keep these limitations with any exploratory comparison.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as e:
        print(f'\nSTOPPED: {e}', file=sys.stderr)
        sys.exit(1)
