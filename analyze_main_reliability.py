"""Echo-EF: read-only analysis of the saved VAL baseline evaluation.

Prints, and only prints:
  1. source-identity checks (checkpoint hash prefix, split membership, saved metrics)
  2. EF error distribution, including the finite-sample 90% quantile half-width
  3. association between |EF head - volume-implied EF| and |EF error|
     (Spearman correlation and top-quintile MAE contrast, each with a
     bootstrap 95% interval)
  4. three-group correctness: pooled counts/accuracy, then an exploratory
     probability estimator for "predicted group is correct" fitted on the
     protocol's recorded calibration positions and evaluated on the
     selection positions, compared against constant and per-group-frequency
     baselines. Fitting only happens if the split membership is recovered
     exactly from saved protocol artifacts; otherwise pooled counts only.
  5. stability appendix (4d): combined fit/eval resampling with full refits
     and paired candidate comparisons; descriptive coefficient spread. No
     automatic conclusion follows from it.

Default run is read-only. Only with the explicit flag
    python .\analyze_main_reliability.py --export-estimate
the script additionally WRITES one new local file, group_correctness_estimate.json,
freezing the reviewed c2 fit and its source identities for the app. Nothing is
written without that flag, and nothing is written if split recovery fails.

No training, no writes, no app or result changes. VAL was used during
development, so every number here is a development-time description of saved
outputs, not a validated guarantee for new videos.

Run from the project root (echo-ef-project):
    python .\\analyze_main_reliability.py
"""
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / 'results' / 'eval_split' / 'val_r3d18_baseline_fp32_k1'
CKPT = ROOT / 'checkpoints' / 'r3d18_best.pt'
LEVEL = 0.90
BOOTSTRAPS = 2000
SEED = 404


def sha256(path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def filelist_csv_path():
    cfg = ROOT / 'configs' / 'base.yaml'
    default = ROOT / 'data' / 'EchoNet-Dynamic' / 'FileList.csv'
    if cfg.is_file():
        m = re.search(r'file_list_csv:\s*["\']?([^"\'\n]+)', cfg.read_text(encoding='utf-8'))
        if m:
            p = Path(m.group(1).strip())
            return p if p.is_absolute() else (ROOT / p)
    return default


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denom) if denom > 0 else float('nan')


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def boot_ci(stat_fn, n, b=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.empty(b)
    for i in range(b):
        vals[i] = stat_fn(rng.integers(0, n, n))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def group_of(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < 40., 0, np.where(x < 50., 1, 2))


GROUP_LABELS = ('Reduced (<40)', 'Mildly reduced (40-<50)', 'Normal (>=50)')
BOUNDARIES = (40., 50.)


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def auc_score(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    r = ranks(p)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def fit_logistic(X, y, alpha=1.0, iters=60, tol=1e-9):
    # Penalized IRLS; fixed L2 on non-intercept coefficients keeps the fit
    # finite under separation. alpha prespecified, not tuned.
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    beta = np.zeros(X.shape[1])
    pen = np.ones(X.shape[1]); pen[0] = 0.0
    for _ in range(iters):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        w = np.maximum(p * (1 - p), 1e-9)
        grad = X.T @ (y - p) - alpha * pen * beta
        H = (X.T * w) @ X + alpha * np.diag(pen)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def predict_logistic(beta, X):
    z = np.asarray(X, dtype=np.float64) @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def design_matrix(d, g, with_group):
    cols = [np.ones(len(d)), d]
    if with_group:
        cols.append((g == 1).astype(np.float64))
        cols.append((g == 2).astype(np.float64))
    return np.column_stack(cols)


def section4_group_correctness(ef, true_ef, names, n):
    print('\n== 4. Three-group correctness (exploratory probability estimator) ==')
    print('Groups derived from EF exactly as the app derives them: <40, 40-<50, >=50.')
    g_true = group_of(true_ef)
    g_pred = group_of(ef)
    correct = (g_true == g_pred).astype(np.float64)

    print('\n-- 4a. Pooled descriptive counts (all saved VAL rows; not a fitted model) --')
    for g in range(3):
        print(f'  true {GROUP_LABELS[g]}: {int((g_true == g).sum())} videos; predicted {GROUP_LABELS[g]}: {int((g_pred == g).sum())}')
    print('  confusion (rows=true group, cols=predicted group, order Reduced/Mildly/Normal):')
    conf = np.zeros((3, 3), dtype=int)
    for t in range(3):
        for pr in range(3):
            conf[t, pr] = int(((g_true == t) & (g_pred == pr)).sum())
    for t in range(3):
        print(f'    true {GROUP_LABELS[t]}: {conf[t].tolist()}')
    acc = float(correct.mean())
    lo, hi = wilson_ci(int(correct.sum()), n)
    print(f'  pooled three-group accuracy: {acc*100:.1f}% (Wilson 95% interval [{lo*100:.1f}, {hi*100:.1f}])')
    for g in range(3):
        m = g_pred == g
        k = int(correct[m].sum()); cnt = int(m.sum())
        lo, hi = wilson_ci(k, cnt)
        if cnt:
            print(f'  when predicting {GROUP_LABELS[g]}: correct {k}/{cnt} = {k/cnt*100:.1f}% (Wilson 95% [{lo*100:.1f}, {hi*100:.1f}])')
        else:
            print(f'  when predicting {GROUP_LABELS[g]}: never predicted on saved VAL')

    print('\n-- 4b. Recovering the protocol split membership from saved artifacts --')
    protocol_path = ROOT / 'results' / 'fourclass_fp32_v3' / 'protocol.json'
    fail = None
    sel_mask = cal_mask = None
    while True:  # single pass; break = fail closed
        if not protocol_path.is_file():
            fail = f'{protocol_path} not found'; break
        try:
            protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        except Exception as e:
            fail = f'protocol.json unreadable: {e}'; break
        sel_pos = protocol.get('selection_positions')
        cal_pos = protocol.get('calibration_positions')
        if not isinstance(sel_pos, list) or not isinstance(cal_pos, list):
            fail = 'selection_positions/calibration_positions missing from protocol.json'; break
        csv_path = filelist_csv_path()
        if not csv_path.is_file():
            fail = f'{csv_path} not found; cannot map positions to videos'; break
        recorded_sha = str(protocol.get('file_list_sha256', ''))
        actual_sha = sha256(csv_path)
        print(f'  FileList.csv sha256[:16] computed={actual_sha[:16]} recorded-in-protocol={recorded_sha[:16]} match={actual_sha == recorded_sha}')
        if actual_sha != recorded_sha:
            fail = 'FileList.csv differs from the file the protocol positions were built on'; break
        with csv_path.open(newline='', encoding='utf-8') as f:
            val_rows = [r for r in csv.DictReader(f) if r.get('Split', '').strip().upper() == 'VAL']
        if len(val_rows) != n:
            fail = f'CSV VAL row count {len(val_rows)} != saved evaluation rows {n}'; break
        all_pos = list(sel_pos) + list(cal_pos)
        if sorted(all_pos) != list(range(n)):
            fail = f'protocol positions do not partition 0..{n-1} (selection={len(sel_pos)}, calibration={len(cal_pos)})'; break
        pos_stem = {}
        ok = True
        for pos in all_pos:
            if not isinstance(pos, int) or pos < 0 or pos >= n:
                fail = f'protocol position {pos!r} out of range'; ok = False; break
            pos_stem[pos] = Path(val_rows[pos]['FileName']).stem
        if not ok:
            break
        if len(set(names)) != n:
            fail = 'duplicate file names in saved evaluation; cannot map positions unambiguously'; break
        row_of = {stem: i for i, stem in enumerate(names)}
        sel_stems = {pos_stem[pos] for pos in sel_pos}
        cal_stems = {pos_stem[pos] for pos in cal_pos}
        if set(names) != sel_stems | cal_stems:
            fail = 'protocol-position file names do not match the saved evaluation videos'; break
        sel_mask = np.array([row_of[s] for s in (pos_stem[pos] for pos in sel_pos)], dtype=int)
        cal_mask = np.array([row_of[s] for s in (pos_stem[pos] for pos in cal_pos)], dtype=int)
        print(f'  recovered: selection {len(sel_mask)} rows, calibration {len(cal_mask)} rows, disjoint, covering all {n} saved videos')
        break
    if fail is not None:
        print(f'  split recovery failed: {fail}')
        print('  FAIL CLOSED: pooled descriptive counts above stand, but no estimator is fitted and no fitted performance is reported. Held-out-looking numbers from pooled fitting would be misleading on this reused data.')
        return None

    fit_idx, eval_idx = cal_mask, sel_mask
    print('\n-- 4c. Estimator candidates (fitted on the 645 calibration rows only, evaluated on the 643 selection rows) --')
    print('  Both partitions are development-reused data (selection rows also selected an epoch of a different model). Results are exploratory diagnostics, not unbiased validation.')
    d_all = np.minimum(np.abs(ef - BOUNDARIES[0]), np.abs(ef - BOUNDARIES[1]))
    y_fit, y_eval = correct[fit_idx], correct[eval_idx]
    d_fit, d_eval = d_all[fit_idx], d_all[eval_idx]
    g_fit, g_eval = g_pred[fit_idx], g_pred[eval_idx]

    preds = {}
    # Baseline b0: constant overall fit-half accuracy.
    p0 = float(y_fit.mean())
    preds['b0 constant'] = (np.full(len(y_eval), p0), np.full(len(y_fit), p0))
    # Baseline b1: per-predicted-group frequency from the fit half.
    freq = {g: (float(y_fit[g_fit == g].mean()) if int((g_fit == g).sum()) else float(y_fit.mean())) for g in range(3)}
    preds['b1 per-group frequency'] = (np.array([freq[g] for g in g_eval]), np.array([freq[g] for g in g_fit]))
    # c1: pooled logistic on distance only.
    beta1 = fit_logistic(design_matrix(d_fit, g_fit, False), y_fit)
    preds['c1 logistic(distance)'] = (predict_logistic(beta1, design_matrix(d_eval, g_eval, False)),
                                      predict_logistic(beta1, design_matrix(d_fit, g_fit, False)))
    # c2: logistic on distance + predicted-group indicators (the offered estimator).
    beta2 = fit_logistic(design_matrix(d_fit, g_fit, True), y_fit)
    preds['c2 logistic(distance+group)'] = (predict_logistic(beta2, design_matrix(d_eval, g_eval, True)),
                                            predict_logistic(beta2, design_matrix(d_fit, g_fit, True)))

    print(f'  fit-half rows: {len(y_fit)} (accuracy {y_fit.mean()*100:.1f}%); eval-half rows: {len(y_eval)} (accuracy {y_eval.mean()*100:.1f}%)')
    for g in range(3):
        print(f'  predicted {GROUP_LABELS[g]}: fit {int((g_fit == g).sum())} rows, eval {int((g_eval == g).sum())} rows')
    min_eval = min(int((g_eval == g).sum()) for g in range(3))
    if min_eval < 30:
        print(f'  WARNING: a predicted group has only {min_eval} eval rows; any per-case probability for that group is unsupported and must not be shown as a number.')

    print('\n  eval-half metrics (lower log loss/Brier better, higher AUC better):')
    metrics_eval = {}
    for name, (p_eval, _) in preds.items():
        ll = log_loss(y_eval, p_eval); br = brier(y_eval, p_eval); au = auc_score(y_eval, p_eval)
        metrics_eval[name] = (ll, br, au)
        print(f'    {name}: log loss {ll:.4f}, Brier {br:.4f}, AUC {au:.3f}')

    rng = np.random.default_rng(SEED + 1)
    ne = len(y_eval)
    ll_boot = {name: np.empty(BOOTSTRAPS) for name in preds}
    diff_b0 = np.empty(BOOTSTRAPS); diff_b1 = np.empty(BOOTSTRAPS)
    pe = {name: preds[name][0] for name in preds}
    for b in range(BOOTSTRAPS):
        ii = rng.integers(0, ne, ne)
        for name in preds:
            ll_boot[name][b] = log_loss(y_eval[ii], pe[name][ii])
        diff_b0[b] = ll_boot['b0 constant'][b] - ll_boot['c2 logistic(distance+group)'][b]
        diff_b1[b] = ll_boot['b1 per-group frequency'][b] - ll_boot['c2 logistic(distance+group)'][b]
    for name in preds:
        lo, hi = np.percentile(ll_boot[name], [2.5, 97.5])
        print(f'    {name} log loss bootstrap 95% interval: [{lo:.4f}, {hi:.4f}]')
    lo0, hi0 = np.percentile(diff_b0, [2.5, 97.5])
    lo1, hi1 = np.percentile(diff_b1, [2.5, 97.5])
    print(f'  log-loss improvement of c2 over b0: {float(np.mean(diff_b0)):+.4f} [bootstrap 95% {lo0:+.4f}, {hi0:+.4f}]')
    print(f'  log-loss improvement of c2 over b1: {float(np.mean(diff_b1)):+.4f} [bootstrap 95% {lo1:+.4f}, {hi1:+.4f}]')

    ll_c2 = metrics_eval['c2 logistic(distance+group)'][0]
    supported = (ll_c2 < metrics_eval['b0 constant'][0] and ll_c2 < metrics_eval['b1 per-group frequency'][0]
                 and lo0 > 0 and lo1 > 0)
    print('\n  prespecified adoption rule: c2 is supported only if its eval log loss is below BOTH baselines AND both bootstrap intervals of the improvement exclude 0.')
    if supported:
        print('  VERDICT: c2 beats both baselines on the eval half. A per-case probability estimator is supported ON THIS REUSED DEVELOPMENT DATA ONLY.')
    else:
        print('  VERDICT: c2 is NOT supported over the simple baselines on the eval half. Only the pooled descriptive frequencies in 4a stand; no per-case probability should be shown.')

    print('\n  calibration of c2 on the eval half (10 equal-width bins):')
    p_c2 = pe['c2 logistic(distance+group)']
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        m = (p_c2 >= edges[b]) & (p_c2 < edges[b + 1] if b < 9 else p_c2 <= edges[b + 1])
        cnt = int(m.sum())
        if cnt:
            mp = float(p_c2[m].mean()); mo = float(y_eval[m].mean())
            ece += cnt / ne * abs(mp - mo)
            print(f'    [{edges[b]:.1f},{edges[b+1]:.1f}]: n={cnt} mean predicted {mp:.3f} observed {mo:.3f}')
    print(f'  ECE(10 bins) = {ece:.4f}')

    print('\n  descriptive check of the distance idea on the eval half (not assumed):')
    order = np.argsort(d_eval, kind='mergesort')
    bins = np.array_split(order, 5)
    for i, b in enumerate(bins):
        print(f'    distance quintile {i+1} (d {float(d_eval[b].min()):.2f}..{float(d_eval[b].max()):.2f}): n={len(b)} accuracy {float(y_eval[b].mean())*100:.1f}%')

    print('  Per-group isotonic/per-group curves: only fitted when every predicted group has >=100 fit rows; otherwise skipped to avoid sparse-curve overfitting.')
    if min(int((g_fit == g).sum()) for g in range(3)) < 100:
        print('  skipped: at least one predicted group has fewer than 100 fit rows.')
    print('  Reminder: all numbers in section 4 describe reused development data. A supported estimator here is still an experimental probability for videos like the saved validation set, not a validated confidence score for a new scan, and not a referral rule.')

    print('\n-- 4d. Stability appendix: combined fit+eval resampling (final bounded check) --')
    print(f'  Each of {BOOTSTRAPS} iterations resamples the {len(y_fit)} fit rows AND the {len(y_eval)} eval rows independently, refits every candidate on the resampled fit rows, and scores all candidates on the same resampled eval rows, keeping comparisons paired within an iteration.')
    rng2 = np.random.default_rng(SEED + 2)
    nf_, ne_ = len(y_fit), len(y_eval)
    cand_names = ['b0 constant', 'b1 per-group frequency', 'c1 logistic(distance)', 'c2 logistic(distance+group)']
    ll_all = {nm: np.empty(BOOTSTRAPS) for nm in cand_names}
    imp0 = np.empty(BOOTSTRAPS); imp1 = np.empty(BOOTSTRAPS)
    coefs = np.empty((BOOTSTRAPS, 4))
    win0 = 0; win1 = 0
    for b in range(BOOTSTRAPS):
        fi = rng2.integers(0, nf_, nf_)
        ei = rng2.integers(0, ne_, ne_)
        yf, df_, gf = y_fit[fi], d_fit[fi], g_fit[fi]
        ye, de_, ge = y_eval[ei], d_eval[ei], g_eval[ei]
        p_b0 = float(yf.mean())
        freq_b = {g: (float(yf[gf == g].mean()) if int((gf == g).sum()) else p_b0) for g in range(3)}
        be1 = fit_logistic(design_matrix(df_, gf, False), yf)
        be2 = fit_logistic(design_matrix(df_, gf, True), yf)
        coefs[b] = be2
        ll_all['b0 constant'][b] = log_loss(ye, np.full(ne_, p_b0))
        ll_all['b1 per-group frequency'][b] = log_loss(ye, np.array([freq_b[g] for g in ge]))
        ll_all['c1 logistic(distance)'][b] = log_loss(ye, predict_logistic(be1, design_matrix(de_, ge, False)))
        ll_all['c2 logistic(distance+group)'][b] = log_loss(ye, predict_logistic(be2, design_matrix(de_, ge, True)))
        imp0[b] = ll_all['b0 constant'][b] - ll_all['c2 logistic(distance+group)'][b]
        imp1[b] = ll_all['b1 per-group frequency'][b] - ll_all['c2 logistic(distance+group)'][b]
        win0 += imp0[b] > 0; win1 += imp1[b] > 0
    print('  candidate log loss, combined-resampling 95% intervals:')
    for nm in cand_names:
        lo, hi = np.percentile(ll_all[nm], [2.5, 97.5])
        print(f'    {nm}: [{lo:.4f}, {hi:.4f}]')
    lo0, hi0 = np.percentile(imp0, [2.5, 97.5]); lo1, hi1 = np.percentile(imp1, [2.5, 97.5])
    print(f'  paired c2 improvement over b0: median {float(np.median(imp0)):+.4f}, 95% interval [{lo0:+.4f}, {hi0:+.4f}]; c2 better in {win0 / BOOTSTRAPS * 100:.1f}% of iterations')
    print(f'  paired c2 improvement over b1: median {float(np.median(imp1)):+.4f}, 95% interval [{lo1:+.4f}, {hi1:+.4f}]; c2 better in {win1 / BOOTSTRAPS * 100:.1f}% of iterations')
    print('  c2 coefficient spread across refits (median [2.5..97.5 percentiles]; descriptive, no pass/fail threshold):')
    for j, cn in enumerate(('intercept', 'distance', 'group=Mildly reduced', 'group=Normal')):
        lo, hi = np.percentile(coefs[:, j], [2.5, 97.5])
        print(f'    {cn}: {float(np.median(coefs[:, j])):+.4f} [{lo:+.4f}, {hi:+.4f}]')
    print('  These intervals describe combined fit+eval resampling variability only. Development-time selection (both partitions, including main-checkpoint selection) and any distribution shift to new videos are outside what resampling can measure. No automatic conclusion follows from these ranges.')
    domain = {}
    for g in range(3):
        m = g_fit == g
        domain[GROUP_LABELS[g]] = {'fit_count': int(m.sum()),
                                   'distance_min': float(d_fit[m].min()) if int(m.sum()) else None,
                                   'distance_max': float(d_fit[m].max()) if int(m.sum()) else None}
    return {'coefficients': [float(v) for v in beta2], 'fit_rows': int(len(y_fit)),
            'eval_rows': int(len(y_eval)), 'fit_domain': domain,
            'eval_log_loss': metrics_eval['c2 logistic(distance+group)'][0]}


def export_estimate(estimate):
    # Explicitly requested by --export-estimate: WRITES one new local JSON file.
    # This is the only mode of this script that writes anything.
    if estimate is None:
        print('\nEXPORT SKIPPED: the estimator was not fitted (see section 4b fail-closed reason). No file written.')
        return
    import datetime
    protocol_path = ROOT / 'results' / 'fourclass_fp32_v3' / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    pcfg = protocol.get('config', {}) if isinstance(protocol.get('config'), dict) else {}
    pdata = pcfg.get('data', {}) if isinstance(pcfg.get('data'), dict) else {}
    metrics_path = EVAL_DIR / 'metrics.json'
    npz_path = EVAL_DIR / 'predictions.npz'
    csv_path = filelist_csv_path()
    met = json.loads(metrics_path.read_text(encoding='utf-8'))
    artifact = {
        'kind': 'echo_ef_group_correctness_estimate',
        'version': 2,
        'fitted_on': datetime.date.today().isoformat(),
        'label': 'EXPERIMENTAL: fitted and evaluated on reused development validation data; not validated on new scans; estimates the chance the predicted EF group is correct, not the exact EF; no clinical or referral cutoff.',
        'model': {
            'type': 'logistic_regression',
            'feature': "min(abs(ef-40), abs(ef-50)) plus predicted-group indicators (Mildly reduced, Normal; Reduced is the reference)",
            'boundaries_ef': [40.0, 50.0],
            'group_order': ['Reduced', 'Mildly reduced', 'Normal'],
            'coefficients': {'intercept': estimate['coefficients'][0], 'distance': estimate['coefficients'][1],
                             'group_mildly_reduced': estimate['coefficients'][2], 'group_normal': estimate['coefficients'][3]},
            'penalty_l2': 1.0,
            'inference': {
                'formula': 'p = sigmoid(intercept + distance*d + group_mildly_reduced*I(predicted group is Mildly reduced) + group_normal*I(predicted group is Normal)); Reduced is the reference group',
                'distance_definition': 'd = min(abs(ef-40), abs(ef-50))',
                'distance_units': 'EF points',
                'feature_scaling': 'none (raw EF points; no normalization applied to features)',
                'z_clip': 35.0,
            },
        },
        'fit_domain': {'fit_rows': estimate['fit_rows'], 'eval_rows': estimate['eval_rows'],
                       'per_predicted_group': estimate['fit_domain']},
        'eval_log_loss_development_only': estimate['eval_log_loss'],
        'identities': {
            'checkpoint_sha256': sha256(CKPT),
            'predictions_npz_sha256': sha256(npz_path),
            'metrics_json_sha256': sha256(metrics_path),
            'protocol_json_sha256': sha256(protocol_path),
            'protocol_version': str(protocol.get('version', '')),
            'file_list_csv_sha256': sha256(csv_path) if csv_path.is_file() else None,
            'analysis_script_sha256': sha256(Path(__file__).resolve()),
        },
        'preprocessing': {
            'clips_per_video': met.get('clips_per_video'),
            'clip_selection': 'single centered clip',
            'num_frames': pdata.get('num_frames'),
            'frame_stride': pdata.get('frame_stride'),
            'frame_size': pdata.get('frame_size'),
            'precision': met.get('precision'),
            'tf32_disabled': met.get('tf32_disabled'),
            'normalization': 'mean/std saved inside the checkpoint; verified by the app at load',
            'source': 'results/eval_split/val_r3d18_baseline_fp32_k1/metrics.json and the protocol config',
        },
    }
    out = ROOT / 'group_correctness_estimate.json'
    out.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    print(f'\nEXPORT WROTE {out.name}: frozen c2 fit ({estimate["fit_rows"]} fit rows), one new local file.')
    for g, d in estimate['fit_domain'].items():
        print(f'  predicted {g}: fit_count={d["fit_count"]}, fitted distance range [{d["distance_min"]}, {d["distance_max"]}]')
    print('The app shows the estimate only inside each predicted group\'s fitted distance range, with matching source identities. No private identifiers or file names were written.')


def main():
    print('Echo-EF saved-VAL baseline analysis (read-only).')
    print(f'Python {sys.version.split()[0]}, numpy {np.__version__}')

    metrics_path = EVAL_DIR / 'metrics.json'
    npz_path = EVAL_DIR / 'predictions.npz'
    for p in (metrics_path, npz_path, CKPT):
        if not p.is_file():
            print(f'\nSTOP: missing {p}. Keep the original local evaluation and checkpoint; do not retrain or re-run evaluation for this check.')
            return
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))

    print('\n== 1. Source identity ==')
    digest = sha256(CKPT)
    recorded = str(metrics.get('checkpoint_sha256_16', ''))
    print(f'checkpoint sha256[:16] computed={digest[:16]} recorded={recorded} match={digest[:16] == recorded}')
    print(f'clips_per_video={metrics.get("clips_per_video")} precision={metrics.get("precision")}')
    if digest[:16] != recorded or metrics.get('clips_per_video') != 1 or metrics.get('precision') != 'fp32':
        print('STOP: saved evaluation does not match the expected baseline identity. Not analysing a substituted source.')
        return

    d = np.load(npz_path, allow_pickle=False)
    keys = list(d.keys())
    print(f'predictions.npz arrays: {keys}')
    need = ('y_true', 'y_pred', 'edv_pred', 'esv_pred', 'file_names')
    if any(k not in keys for k in need):
        print('STOP: expected arrays y_true/y_pred/edv_pred/esv_pred/file_names not all present. Send this output back; do not edit the file.')
        return
    ef = np.asarray(d['y_pred'], dtype=np.float64).ravel()
    true_ef = np.asarray(d['y_true'], dtype=np.float64).ravel()
    edv = np.asarray(d['edv_pred'], dtype=np.float64).ravel()
    esv = np.asarray(d['esv_pred'], dtype=np.float64).ravel()
    if not (len(ef) == len(true_ef) == len(edv) == len(esv)):
        print('STOP: saved arrays have different lengths. Send this output back; do not edit the file.')
        return
    try:
        raw_names = d['file_names']
    except ValueError:
        # Older saves stored file_names as an object array; strings only, from the local verified folder.
        raw_names = np.load(npz_path, allow_pickle=True)['file_names']
    names = [Path(str(v)).stem for v in raw_names]
    n = len(ef)
    print(f'videos in saved evaluation: {n}')

    csv_path = filelist_csv_path()
    if csv_path.is_file():
        with csv_path.open(newline='', encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if r.get('Split', '').strip().upper() == 'VAL']
        val_names = {Path(r['FileName']).stem for r in rows}
        same = set(names) == val_names
        print(f'FileList VAL rows: {len(val_names)}; saved names match VAL set exactly: {same}')
        if not same:
            print('STOP: saved predictions are not exactly the configured VAL split. Not analysing a mismatched source.')
            return
    else:
        print(f'note: {csv_path} not found; split membership not re-checked here.')

    mae = float(np.mean(np.abs(ef - true_ef)))
    rmse = float(np.sqrt(np.mean((ef - true_ef) ** 2)))
    ss_res = float(np.sum((ef - true_ef) ** 2))
    ss_tot = float(np.sum((true_ef - true_ef.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f'recomputed from saved predictions: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}')
    for k in sorted(metrics):
        if isinstance(metrics[k], (int, float)):
            print(f'  metrics.json {k} = {metrics[k]}')

    print('\n== 2. EF error distribution (VAL, saved outputs) ==')
    abs_err = np.abs(ef - true_ef)
    finite = np.isfinite(abs_err)
    print(f'finite EF pairs: {int(finite.sum())} of {n}')
    abs_err = abs_err[finite]
    m = len(abs_err)
    srt = np.sort(abs_err)
    k = math.ceil((m + 1) * LEVEL)
    q = float(srt[min(k, m) - 1])
    cov = float(np.mean(abs_err <= q))
    print(f'MAE={mae:.3f}  median |err|={float(np.median(abs_err)):.3f}  RMSE={rmse:.3f}')
    for lvl in (0.50, 0.80, 0.90, 0.95):
        kk = math.ceil((m + 1) * lvl)
        qq = float(srt[min(kk, m) - 1])
        print(f'  finite-sample {int(lvl*100)}% half-width (rank {min(kk, m)} of {m}): {qq:.2f} EF points; observed fraction within +/-{qq:.2f}: {float(np.mean(abs_err <= qq)) * 100:.1f}%')
    print('Same width for every video. VAL was reused during development: this describes saved development outputs, not guaranteed coverage on new videos.')

    print('\n== 3. EF-head vs volume-implied-EF disagreement ==')
    valid = (np.isfinite(ef) & np.isfinite(edv) & np.isfinite(esv)
             & (edv > 0) & (esv >= 0) & (esv <= edv))
    implied = np.full(n, np.nan)
    implied[valid] = 100.0 * (edv[valid] - esv[valid]) / edv[valid]
    plausible = valid & (implied >= 0) & (implied <= 100)
    print(f'rows with nonfinite EDV/ESV, EDV<=0, ESV<0 or ESV>EDV: {int((~valid).sum())}')
    print(f'rows with implied EF outside 0-100: {int((valid & ~plausible).sum())}')
    if int((~plausible).sum()) > 0:
        print('note: implausible rows are excluded from the disagreement analysis only.')
    idx = np.where(plausible)[0]
    dis = np.abs(ef[idx] - implied[idx])
    err = np.abs(ef[idx] - true_ef[idx])
    nv = len(idx)
    print(f'analysed videos: {nv}; disagreement median={float(np.median(dis)):.2f}, p80={float(np.quantile(dis, 0.8)):.2f}, max={float(dis.max()):.2f} EF points')
    rho = spearman(dis, err)
    lo, hi = boot_ci(lambda ii: spearman(dis[ii], err[ii]), nv)
    print(f'Spearman(disagreement, |EF error|) = {rho:.3f}  bootstrap 95% interval [{lo:.3f}, {hi:.3f}]')
    thr = float(np.quantile(dis, 0.8))
    top = dis >= thr
    mae_top = float(np.mean(err[top]))
    mae_rest = float(np.mean(err[~top]))
    diff = mae_top - mae_rest
    lo2, hi2 = boot_ci(lambda ii: float(np.mean(err[ii][dis[ii] >= thr])) - float(np.mean(err[ii][dis[ii] < thr])), nv)
    print(f'top-20% disagreement (d>={thr:.2f}): {int(top.sum())} videos, MAE={mae_top:.2f}; remaining MAE={mae_rest:.2f}; difference={diff:+.2f} [{lo2:+.2f}, {hi2:+.2f}]')
    print('Exploratory, same-VAL association. The top-20% split is descriptive, not a validated or clinical cutoff. Association here alone does not accept or reject an app warning.')

    estimate = section4_group_correctness(ef, true_ef, names, n)

    if '--export-estimate' in sys.argv[1:]:
        export_estimate(estimate)
    else:
        print('\nRead-only analysis complete. No files were created or modified.')


if __name__ == '__main__':
    main()
