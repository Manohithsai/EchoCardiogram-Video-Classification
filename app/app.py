"""Local Echo-EF app. Run from project root: python app/app.py
One new entry point; no training, recalibration, result overwrite or public sharing.
Only use de-identified EchoNet-style apical-four-chamber echo clips.
"""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
# Support the requested root app/app.py and the earlier module entry point.
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src/models/r3d_multitask.py").is_file()), None)
if ROOT is None:
    raise SystemExit("Place this file inside echo-ef-project/app, beside the existing src folder.")
sys.path.insert(0, str(ROOT))
from src.infer import predict_video as pv
from src.models.r3d_multitask import R3DMultiTask
from src.models.swin3d_multitask import SwinMultiTask

R3D = 'R3D baseline'
SWIN = 'Swin baseline'
RESIDUAL = 'Residual-error research model'
OPTIONS = [R3D, RESIDUAL]
CHECKPOINTS = {R3D: 'checkpoints/r3d18_best.pt', SWIN: 'checkpoints/swin3d_t_best.pt',
               RESIDUAL: 'checkpoints/r3d18_mcdropout_best.pt'}
REFERENCES = {R3D: 'results/eval_split/val_r3d18_baseline_fp32_k1/metrics.json',
              SWIN: 'results/eval_split/val_swin3d_t_baseline_fp32_k1/metrics.json'}
CALIBRATION = 'results/uncertainty_calibration/residual_val_calibration.json'
HEAD = 'checkpoints/r3d18_residual_uncertainty_best.pt'
MAX_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 4096  # explicit app resource limit, not a validated clinical boundary
LOCK = threading.Lock()
CACHE = {}  # one active model only, no patient/video cache


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def decode(path):
    p = Path(path)
    if p.suffix.lower() not in ('.avi', '.mp4'):
        raise ValueError('Use an AVI or MP4 echo video. DICOM, images and arbitrary files are not supported.')
    if not p.is_file() or not 0 < p.stat().st_size <= MAX_BYTES:
        raise ValueError('Choose a nonempty video no larger than 256 MiB.')
    cap = cv2.VideoCapture(str(p))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError('Could not open this video. Check its codec or export a de-identified AVI.')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= MAX_FRAMES:
                raise ValueError('Video exceeds the app limit of 4096 frames. Use a shorter echo clip.')
            # Same grayscale + INTER_AREA resize as build_cache and predict_video.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise ValueError('No frames decoded. This codec or file may be unsupported or damaged.')
    return np.stack(frames).astype(np.uint8)


def validate_norm(mean, std):
    m, s = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if m.shape != (3,) or s.shape != (3,) or not np.isfinite(m).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('Invalid saved normalization; prediction stopped.')
    return m.tolist(), s.tolist()


def check_settings(ckpt):
    for source in (ckpt, ckpt.get('config', {}), ckpt.get('cfg', {})):
        if not isinstance(source, dict):
            continue
        for d in (source, source.get('data', {})):
            for name, expected in (('num_frames', 32), ('frame_stride', 2), ('frame_size', 112)):
                if name in d and int(d[name]) != expected:
                    raise ValueError(f'Checkpoint {name} differs from the supported 32/2/112 recipe.')


def checkpoint(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict) or 'model_state_dict' not in obj:
        raise ValueError('Unsupported checkpoint format. No unsafe loading or guessed architecture is allowed.')
    check_settings(obj)
    return obj


def calibration_status(device):
    """Fail closed for ranges; never refit or mutate the calibration artifact."""
    notes = []
    try:
        cal = read_json(CALIBRATION)
        expected = {'num_frames': 32, 'frame_stride': 2, 'frame_size': 112,
                    'mean': pv.R3D_MEAN, 'std': pv.R3D_STD, 'clips_per_video': 1,
                    'autocast': False, 'logvar_clamp': [-6.0, 6.0], 'score_sigma_floor': .001}
        if any(cal['settings'].get(k) != v for k, v in expected.items()):
            raise ValueError('Calibration preprocessing/settings do not match.')
        for k, p in (('mean_checkpoint_sha256', CHECKPOINTS[RESIDUAL]),
                     ('head_checkpoint_sha256', HEAD), ('config_sha256', 'configs/base.yaml')):
            if sha(ROOT / p) != cal[k]:
                raise ValueError(f'Calibration fingerprint mismatch: {p}.')
        if not (0 < float(cal['alpha']) < 1 and int(cal['n_val']) > 0 and math.isfinite(float(cal['q_hat'])) and float(cal['q_hat']) > 0):
            raise ValueError('Invalid calibration values.')
        # Exact computational source identity is necessary for an old calibrated range.
        for p in ('src/infer/predict_video.py', 'src/infer/predict_uncertainty.py',
                  'src/models/r3d_multitask.py', 'src/models/r3d_residual_uncertainty.py',
                  'src/data/echonet_dataset.py', 'src/eval/residual_uncertainty_eval.py'):
            raw=(ROOT / p).read_bytes()
            lf=raw.replace(b'\r\n',b'\n')
            variants={hashlib.sha256(b).hexdigest() for b in (raw,lf,lf.replace(b'\n',b'\r\n'))}
            if cal.get('source_sha256', {}).get(p) not in variants:
                raise ValueError(f'Source compatibility unresolved beyond line endings: {p}.')
        from src.infer.predict_uncertainty import runtime_info
        now = runtime_info(device)
        for k, v in now.items():
            if v != cal['runtime'].get(k):
                notes.append(f'Runtime differs: {k}.')
        if notes:
            notes.append('Runtime differs from calibration; exact numerical equivalence is not established. Interpret this development-time range cautiously.')
        return cal, notes
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, [str(exc), 'Range unavailable; no recalibration or model substitution performed.']


def load_model(choice, device):
    path = ROOT / CHECKPOINTS[choice]
    if not path.is_file():
        raise ValueError(f'Missing {CHECKPOINTS[choice]}. Keep the original local checkpoint; do not retrain for this app.')
    fingerprint = sha(path)
    key = (choice, fingerprint, str(device))
    if CACHE.get('key') == key:
        return CACHE['value']
    CACHE.clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    ckpt = checkpoint(path)
    notes = []
    if choice == SWIN:
        if 'norm_mean' not in ckpt or 'norm_std' not in ckpt:
            raise ValueError('Swin checkpoint needs its saved normalization. No guessed ImageNet fallback.')
        mean, std = validate_norm(ckpt['norm_mean'], ckpt['norm_std'])
        model = SwinMultiTask(pretrained=False)
    else:
        mean, std = validate_norm(pv.R3D_MEAN, pv.R3D_STD)
        if 'norm_mean' in ckpt or 'norm_std' in ckpt:
            saved_m, saved_s = validate_norm(ckpt.get('norm_mean'), ckpt.get('norm_std'))
            if not np.allclose(saved_m, mean, rtol=0, atol=1e-7) or not np.allclose(saved_s, std, rtol=0, atol=1e-7):
                raise ValueError('R3D normalization differs from the verified single-video path.')
        model = R3DMultiTask(pretrained=False, dropout_p=pv.DROPOUT_P)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    if choice in REFERENCES:
        ref = read_json(REFERENCES[choice])
        if (ref.get('checkpoint_sha256_16') != fingerprint[:16] or ref.get('clips_per_video') != 1
                or ref.get('precision') != 'fp32'):
            raise ValueError('Checkpoint does not match the recorded baseline evaluation. No silent substitution.')
        rm, rs = validate_norm(ref['normalisation_mean'], ref['normalisation_std'])
        if not np.allclose(mean, rm) or not np.allclose(std, rs):
            raise ValueError('Normalization does not match the saved evaluation.')
    else:
        cal = read_json(CALIBRATION)
        if fingerprint != cal.get('mean_checkpoint_sha256'):
            raise ValueError('Residual mean checkpoint does not match the saved model-pair identity.')
        notes.append('Different mean checkpoint from R3D baseline. This is deterministic inference, not MC dropout.')
    value = model, mean, std, fingerprint, ckpt.get('epoch'), notes
    CACHE.update(key=key, value=value)
    return value


GROUP_ESTIMATE_CACHE = 'not-loaded'

GROUP_ESTIMATE_REASONS = {
    'missing_or_unverified': 'Group-correctness estimate unavailable: the saved fit file is missing or could not be verified.',
    'mismatch': "Group-correctness estimate unavailable: the saved fit does not match this app's verified model and pipeline.",
    'outside_domain': 'Group-correctness estimate not shown: this result is outside the range the estimate was fitted on.',
}


def group_estimate_artifact():
    """Lazy-load the frozen experimental estimator artifact; any problem marks it unusable with a reason."""
    global GROUP_ESTIMATE_CACHE
    if GROUP_ESTIMATE_CACHE != 'not-loaded':
        return GROUP_ESTIMATE_CACHE
    art = None
    reason = 'missing_or_unverified'
    try:
        a = json.loads((ROOT / 'group_correctness_estimate.json').read_text(encoding='utf-8'))
        if a.get('kind') != 'echo_ef_group_correctness_estimate' or a.get('version') != 2:
            raise ValueError('unsupported estimate artifact')
        m = a['model']
        if m.get('type') != 'logistic_regression' or m.get('boundaries_ef') != [40.0, 50.0] or m.get('group_order') != ['Reduced', 'Mildly reduced', 'Normal']:
            raise ValueError('estimate artifact group definition differs from the reviewed estimator')
        c = m['coefficients']
        coefs = (float(c['intercept']), float(c['distance']), float(c['group_mildly_reduced']), float(c['group_normal']))
        if not all(math.isfinite(v) for v in coefs):
            raise ValueError('nonfinite estimate coefficients')
        # Inference-formula parity: the artifact must describe exactly what is implemented below.
        inf = m.get('inference', {})
        if (inf.get('distance_definition') != 'd = min(abs(ef-40), abs(ef-50))'
                or inf.get('distance_units') != 'EF points'
                or 'Reduced' not in str(inf.get('formula', ''))
                or not str(inf.get('feature_scaling', '')).startswith('none')
                or float(inf.get('z_clip')) != 35.0):
            raise ValueError('estimate artifact inference formula differs from the implemented computation')
        # Preprocessing identity: the fit assumed this exact pipeline.
        pre = a.get('preprocessing', {})
        if (pre.get('clips_per_video') != 1 or pre.get('clip_selection') != 'single centered clip'
                or pre.get('num_frames') != 32 or pre.get('frame_stride') != 2
                or pre.get('frame_size') != 112 or pre.get('precision') != 'fp32' or pre.get('tf32_disabled') is not True):
            reason = 'mismatch'
            raise ValueError('estimate artifact preprocessing differs from this pipeline')
        dom = a['fit_domain']['per_predicted_group']
        domains = {}
        for g in ('Reduced (<40)', 'Mildly reduced (40-<50)', 'Normal (>=50)'):
            d = dom[g]
            n_fit, dmin, dmax = int(d['fit_count']), float(d['distance_min']), float(d['distance_max'])
            if n_fit < 1 or not (math.isfinite(dmin) and math.isfinite(dmax)) or dmin < 0 or dmax < dmin:
                raise ValueError('invalid fitted domain')
            domains[g] = (n_fit, dmin, dmax)
        ids = a.get('identities', {})
        art = {'coefs': coefs, 'domains': domains,
               'checkpoint_sha256': str(ids.get('checkpoint_sha256', '')),
               'metrics_json_sha256': str(ids.get('metrics_json_sha256', ''))}
    except (OSError, KeyError, TypeError, ValueError):
        art = None
    GROUP_ESTIMATE_CACHE = {'art': art, 'reason': None if art is not None else reason}
    return GROUP_ESTIMATE_CACHE


def group_correctness_estimate(ef, choice, fingerprint):
    """Frozen reviewed estimator for the R3D baseline only, strictly inside its fitted distance domain."""
    if choice != R3D:
        return None
    loaded = group_estimate_artifact()
    art = loaded['art']
    if art is None:
        return {'status': 'unavailable', 'reason': loaded['reason']}
    if (not art['checkpoint_sha256'] or fingerprint != art['checkpoint_sha256']
            or not art['metrics_json_sha256'] or sha(ROOT / REFERENCES[R3D]) != art['metrics_json_sha256']):
        return {'status': 'unavailable', 'reason': 'mismatch'}
    group = 'Reduced' if ef < 40 else ('Mildly reduced' if ef < 50 else 'Normal')
    key = {'Reduced': 'Reduced (<40)', 'Mildly reduced': 'Mildly reduced (40-<50)', 'Normal': 'Normal (>=50)'}[group]
    _, dmin, dmax = art['domains'][key]
    d = min(abs(ef - 40.0), abs(ef - 50.0))
    if d < dmin or d > dmax:
        return {'status': 'outside_domain', 'group': group}
    b0, b1, bm, bn = art['coefs']
    z = b0 + b1 * d + (bm if group == 'Mildly reduced' else 0.0) + (bn if group == 'Normal' else 0.0)
    p = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
    return {'status': 'ok', 'group': group, 'pct': int(round(100 * p))}


def result_card(values=None, range_info=None, choice=None, estimate=None):
    import html
    if values is None:
        return '<div class="result-empty"><span class="eyebrow">YOUR RESULT</span><h2>See the video.<br>Understand the estimate.</h2><p>Choose a local TEST sample, then select Analyze video. The sampled frame and model result appear together.</p><div class="empty-mark">EF <span>%</span></div></div>'
    ef = values['ef']
    group = 'Reduced' if ef < 40 else ('Mildly reduced' if ef < 50 else 'Normal')
    explanation = {'Reduced':'The estimated EF is below 40%.', 'Mildly reduced':'The estimated EF is between 40% and below 50%.', 'Normal':'The estimated EF is 50% or above.'}[group]
    marker = min(98,max(2,ef))
    cells = ''.join(f'<div class="group-cell {"selected" if name==group else ""}"><strong>{name}</strong><span>{label}</span></div>' for name,label in [('Reduced','Below 40%'),('Mildly reduced','40% to below 50%'),('Normal','50% or above')])
    est_block = ''
    if estimate and estimate['status'] == 'ok':
        mild = '<p class="est-note">Mildly reduced was the weakest group in validation.</p>' if estimate['group'] == 'Mildly reduced' else ''
        est_block = f'<div class="group-estimate"><div><span class="estimate-title">GROUP CONFIDENCE</span><div class="estimate-pct">{estimate["pct"]}<span>%</span></div><span class="estimate-kind">EXPERIMENTAL</span></div><div><strong>Estimated chance the predicted EF group is correct.</strong><p>Based on reused validation data; not validated on new scans. This is about the group label, not the exact EF value.</p>{mild}</div></div>'
    elif estimate and estimate['status'] in ('outside_domain', 'unavailable'):
        reason = GROUP_ESTIMATE_REASONS[estimate['status'] if estimate['status'] == 'outside_domain' else estimate['reason']]
        est_block = f'<div class="estimate-unavailable"><strong>Experimental group estimate unavailable</strong><p>{html.escape(reason)}</p></div>'
    identity = html.escape(choice or 'Predictor not specified')
    separate = '<p class="separate-model">Separate residual-error predictor. Its EF and any paired range do not belong to the main baseline. See Research for the experiment and Technical details for provenance.</p>' if choice == RESIDUAL else ''
    return f'<div class="result-card"><div class="result-identity">{identity}</div>{separate}<span class="eyebrow">ESTIMATED EJECTION FRACTION</span><div class="ef-number">{ef:.1f}<span>%</span></div><div class="group-title">{group}</div><p class="group-summary">{explanation} This group is derived from the estimated EF.</p><div class="ef-scale"><i style="left:{marker}%"></i></div><div class="group-grid">{cells}</div>{est_block}<div class="volume-row"><div><span>Before contraction · EDV</span><strong>{values["edv"]:.1f} <small>mL</small></strong></div><div><span>After contraction · ESV</span><strong>{values["esv"]:.1f} <small>mL</small></strong></div></div><p class="ef-definition">EF estimates the fraction of blood pumped out with each beat. Volumes are predicted separately and need not agree exactly with EF.</p></div>'


def run_prediction(path, choice, device_choice):
    """Serialized inference prevents model/precision state races on the laptop GPU."""
    if not path:
        return '<div class="result-empty"><h2>Choose a TEST sample first.</h2><p>Choose a video from the local TEST list, then select Analyze video.</p></div>', None, {}, 'No TEST sample selected.'
    try:
        if choice not in OPTIONS:
            raise ValueError('Select one of the supported models.')
        with LOCK:
            t0 = time.perf_counter()
            if device_choice == 'GPU' and not torch.cuda.is_available():
                raise ValueError('CUDA GPU is unavailable. Main analysis uses Auto device selection; restart the app and try again.')
            device = torch.device('cuda' if device_choice != 'CPU' and torch.cuda.is_available() else 'cpu')
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            model, mean, std, fingerprint, epoch, model_notes = load_model(choice, device)
            arr = decode(path)
            idx = pv.centred_clip_indices(len(arr))
            clip = torch.from_numpy(arr[idx].astype(np.float32) / 255.).unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0).to(device)
            mu = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
            sd = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                out = model((clip - mu) / sd)
            values = {k: float(out[k].item()) for k in ('ef', 'edv', 'esv')}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Model returned nonfinite outputs. No prediction is reported.')
            notes = list(model_notes)
            if not 0 <= values['ef'] <= 100 or values['edv'] <= 0 or values['esv'] < 0 or values['esv'] > values['edv']:
                raise ValueError('Model output is outside physiological bounds. Prediction withheld, not clipped or repaired. Try a supported de-identified echo clip.')
            range_info = None
            if choice == RESIDUAL:
                cal, cal_notes = calibration_status(device)
                notes.extend(cal_notes)
                if cal is not None:
                    from src.infer.predict_uncertainty import load_pair, predict_one
                    pair, _, _ = load_pair(device)
                    ef, sigma = predict_one(pair, arr, device)
                    if abs(ef - values['ef']) > .001:
                        raise ValueError('Residual pair mean differs from the displayed EF. Range not attached.')
                    half = float(cal['q_hat']) * sigma
                    range_info = {'low': ef - half, 'high': ef + half, 'sigma': sigma,
                                  'nominal_level': 1 - float(cal['alpha']), 'n_val': cal['n_val']}
                    del pair
            group = pv.ef_group(values['ef'])
            text = f"## {values['ef']:.1f}%\nEstimated ejection fraction\n\n**EF group: {group}**\n\nEDV {values['edv']:.1f} mL · ESV {values['esv']:.1f} mL"
            if range_info:
                text += f"\n\nDevelopment-time {100*range_info['nominal_level']:.0f}% nominal range: **{range_info['low']:.1f} to {range_info['high']:.1f} EF points**. Selection-dependent VAL calibration; not a clinical confidence interval or coverage guarantee."
            elif choice == RESIDUAL:
                text += '\n\nResearch range unavailable: calibration compatibility could not be verified. See Technical details.'
            if range_info and notes:
                text += '\n\nRuntime/provenance notes apply to this research range. See Technical details.'
            text += '\n\nConfidence: no calibrated class probability is available for these EF-derived groups.'
            details = {'model': choice, 'checkpoint': CHECKPOINTS[choice], 'checkpoint_sha256': fingerprint,
                       'checkpoint_epoch': epoch, 'outputs': values, 'ef_group': group,
                       'group_definition': '<40 / 40 to <50 / >=50; derived from EF, not class probabilities',
                       'decoded_frames': len(arr), 'sampled_first_last': [int(idx[0]), int(idx[-1])],
                       'sampled_indices_zero_based': idx.tolist(),
                       'input_sha256': sha(path),
                       'volume_implied_ef': 100 * (values['edv'] - values['esv']) / values['edv'],
                       'confidence': 'No calibrated class probability for EF-derived groups.',
                       'sampling': 'single centered 32-frame clip, stride2; short videos loop-padded',
                       'normalization_mean': mean, 'normalization_std': std, 'precision': 'FP32, TF32 off, eval mode',
                       'device': str(device), 'seconds': round(time.perf_counter() - t0, 2),
                       'range': range_info, 'notes': notes,
                       'limits': 'Trained on EchoNet-style videos. No view/quality detector or clinical validation. May be wrong even within physiological bounds.'}
            details['display_group'] = 'Reduced' if values['ef'] < 40 else ('Mildly reduced' if values['ef'] < 50 else 'Normal')
            estimate = group_correctness_estimate(values['ef'], choice, fingerprint)
            if estimate and estimate['status'] == 'ok':
                details['confidence'] = f"Experimental estimated chance the predicted EF group is correct: {estimate['pct']}% (frozen logistic estimator, reused development validation data; not validated on new scans)."
            elif estimate and estimate['status'] in ('outside_domain', 'unavailable'):
                details['confidence'] = GROUP_ESTIMATE_REASONS[estimate['status'] if estimate['status'] == 'outside_domain' else estimate['reason']]
            details['group_correctness_estimate'] = estimate
            return result_card(values, range_info, choice, estimate), arr[int(idx[len(idx)//2])], details, 'Analysis complete. Review the frame and result together.'
    except (Exception, SystemExit) as exc:
        # Always clear ALL result components, so old outputs cannot masquerade as a new run.
        message = 'Prediction stopped: ' + str(exc)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            message = 'GPU memory is full. Close other GPU work, then retry. No result is reported.'
        return '<div class="result-empty"><h2>No result reported.</h2><p>Check the status message and try again.</p></div>', None, {'error': message}, message


def available_test_samples():
    """Current configured TEST rows with original raw files; no outcome-based filtering."""
    import csv
    import yaml
    config = yaml.safe_load((ROOT/'configs/base.yaml').read_text(encoding='utf-8'))['data']
    def resolve(key):
        p = Path(config[key]).expanduser()
        return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
    directory, csv_path = resolve('raw_video_dir'), resolve('file_list_csv')
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        if not {'FileName','Split','EF'}.issubset(reader.fieldnames or []):
            raise ValueError('Configured FileList.csv lacks FileName, Split or EF.')
        rows = list(reader)
    counts = {}
    for row in rows:
        name = row['FileName'].strip()
        counts[name] = counts.get(name,0)+1
    found = {}
    excluded = 0
    for row in rows:
        if row['Split'].strip().upper() != 'TEST':
            continue
        name = row['FileName'].strip()
        if not name or Path(name).name != name or '/' in name or '\\' in name or counts[name] != 1:
            excluded += 1
            continue
        path = (directory/(name if Path(name).suffix else name+'.avi')).resolve()
        if path.parent != directory or not path.is_file():
            excluded += 1
            continue
        found[name] = str(path)
    return dict(sorted(found.items())), excluded


def refresh_test_samples():
    import gradio as gr
    try:
        found, missing = available_test_samples()
        return gr.update(choices=list(found),value=None), None, f'{len(found)} local TEST samples available. {missing} TEST entries unavailable or ambiguous. Sorted by filename, not by model result. Select a sample to begin.'
    except Exception as exc:
        return gr.update(choices=[],value=None), None, 'Test samples unavailable: '+str(exc)+'. Check the dataset paths in configs/base.yaml; no training or download was started.'


def select_test_sample(name):
    try:
        found, excluded = available_test_samples()
        if name is None:
            return None, f'{len(found)} local TEST samples available; {excluded} TEST entries missing or ambiguous. Select a sample to begin.'
        if name not in found:
            return None, 'Choose an available TEST sample. Refresh the list if files changed.'
        return found[name], 'TEST sample selected. Analyze it to see the prediction; the recorded EF is not used by the model.'
    except Exception as exc:
        return None, 'Test sample unavailable: '+str(exc)


def test_comparison(path, prediction):
    """Read labels ONLY after inference. Exact bytes must match a local configured TEST source."""
    unavailable = '### Actual EF unavailable\n'
    try:
        if not path or not prediction or 'outputs' not in prediction:
            return unavailable+'Choose a TEST sample and analyze it first.'
        upload_hash = sha(path)
        if upload_hash != prediction.get('input_sha256'):
            return unavailable+'The input changed. Analyze it again before comparing.'
        import yaml
        import csv
        config = yaml.safe_load((ROOT/'configs/base.yaml').read_text(encoding='utf-8'))
        data = config['data']
        def configured(name):
            value = Path(data[name]).expanduser()
            return value if value.is_absolute() else ROOT/value
        csv_path = configured('file_list_csv')
        raw_dir = configured('raw_video_dir').resolve()
        csv_hash = sha(csv_path)
        with csv_path.open(encoding='utf-8-sig',newline='') as f:
            reader = csv.DictReader(f)
            if not {'FileName','Split','EF'}.issubset(reader.fieldnames or []):
                raise ValueError('Configured FileList.csv lacks required FileName, Split or EF columns.')
            # The name is a candidate lookup only, never the verification itself.
            stem = Path(path).stem
            rows = [r for r in reader if Path(r['FileName']).stem == stem]
        if len(rows) != 1:
            return unavailable+'No unique dataset entry matches this filename. Renamed or outside videos cannot be verified here. Use an unchanged local EchoNet TEST AVI with its original filename.'
        row = rows[0]
        if row['Split'].strip().upper() != 'TEST':
            return unavailable+'The matching entry is not in the official TEST split. This tab compares TEST samples only.'
        name = row['FileName'].strip()
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Dataset filename is not a plain local filename.')
        source = (raw_dir/(name if Path(name).suffix else name+'.avi')).resolve()
        if source.parent != raw_dir or not source.is_file():
            return unavailable+'The original raw video is missing from the configured dataset directory. A filename or cached clip alone is not enough to verify identity.'
        if source.stat().st_size != Path(path).stat().st_size or sha(source) != upload_hash:
            return unavailable+'The uploaded bytes do not match the original configured dataset video. No label is attached to this result.'
        actual = float(row['EF'])
        if not math.isfinite(actual) or not 0 <= actual <= 100:
            raise ValueError('Recorded EF is invalid.')
        if sha(csv_path) != csv_hash or sha(path) != upload_hash:
            raise ValueError('The label file or uploaded input changed during verification. Retry.')
        estimated = float(prediction['outputs']['ef'])
        def group(ef):
            return 'Reduced' if ef < 40 else ('Mildly reduced' if ef < 50 else 'Normal')
        actual_group, estimated_group = group(actual), group(estimated)
        error = abs(estimated-actual)
        agreement = 'Same EF group' if actual_group == estimated_group else 'Different EF groups'
        return (f'### Actual EF: {actual:.2f}% · Predicted EF: {estimated:.2f}%\n\n'
                f'**Absolute error: {error:.2f} EF percentage points**\n\n'
                f'| | Recorded reference | Model prediction |\n|---|---|---|\n'
                f'| EF group | {actual_group} | {estimated_group} |\n\n'
                f'**{agreement}.** Groups use <40%, 40% to below 50%, and >=50%. '
                'A group match does not mean the EF estimate is exact. One sample cannot establish overall model accuracy.\n\n'
                f'Predictor: **{prediction["model"]}**. Reference: EF in configured local FileList.csv, official TEST split. '
                'Identity verified by SHA-256 equality with the original raw video, not its filename alone. '
                'The label was read after prediction and was not passed to the model.\n\n'
                f'Local label-file SHA-256: `{csv_hash}`')
    except (Exception, SystemExit) as exc:
        return unavailable+str(exc)+' No label has been substituted or inferred.'


def check_uncertainty(path, device_choice, prediction):
    """A separate result, never a range around the currently selected baseline."""
    if not prediction or 'outputs' not in prediction:
        return 'Analyze a TEST sample first, then run the residual-error experiment.'
    try:
        if sha(path) != prediction.get('input_sha256'):
            return 'The video changed. Analyze it again before running the experiment.'
        _, _, details, status = run_prediction(path, RESIDUAL, device_choice)
        if 'outputs' not in details:
            return 'Uncertainty unavailable. '+status+' The main result has not changed.'
        interval = details.get('range')
        if interval is None:
            return 'Uncertainty range unavailable: the saved model pair or calibration could not be verified. '+ ' '.join(details.get('notes',[]))+' The main result has not changed.'
        ef = details['outputs']['ef']
        level = 100*interval['nominal_level']
        low, high = interval['low'], interval['high']
        notes = ' '.join(details.get('notes',[]))
        return (f'### Separate uncertainty estimate: {ef:.1f}% EF\n'
                f'**Prediction range: {low:.1f}% to {high:.1f}% EF**\n\n'
                f'Range width: {high-low:.1f} EF percentage points. A wider range means this predictor is less precise. '
                'It does not tell us whether the scan is normal or whether medical review is unnecessary.\n\n'
                f'This is a **different predictor** from the main R3D baseline. Its range belongs only to its own {ef:.1f}% EF estimate above, not to the main result. '
                f'The saved range has a {level:.0f}% nominal level, fitted on {interval["n_val"]} validation videos also used during model development. '
                'That level is not a per-video confidence percentage or a coverage guarantee. No validated low/high-confidence cutoff is available. '
                'The limits are not clipped to 0-100; crossing a group boundary does not give group probabilities.\n\n'
                '**Consult a qualified doctor to interpret the scan.**\n\n'
                f'Model/calibration notes: {notes or "Saved fingerprints and settings matched."}')
    except (Exception, SystemExit) as exc:
        return 'Uncertainty unavailable: '+str(exc)+'. The main result has not changed.'


def run_explanation(path, choice, device_choice, prediction):
    """Same model/input as the visible prediction. No retained autograd hooks or video state."""
    try:
        if not prediction or 'outputs' not in prediction:
            raise ValueError('Analyze the current video first.')
        if prediction['model'] != choice or sha(path) != prediction['input_sha256']:
            raise ValueError('Video or model changed. Analyze again before explaining.')
        with LOCK:
            device = torch.device(prediction['device'])
            model, mean, std, fingerprint, _, _ = load_model(choice, device)
            if fingerprint != prediction['checkpoint_sha256']:
                raise ValueError('Checkpoint changed. Analyze again.')
            arr = decode(path)
            idx = pv.centred_clip_indices(len(arr))
            x = torch.from_numpy(arr[idx].astype(np.float32)/255.).unsqueeze(0).repeat(3,1,1,1).unsqueeze(0).to(device)
            x = ((x-torch.tensor(mean,device=device).view(1,3,1,1,1))/torch.tensor(std,device=device).view(1,3,1,1,1))
            captured = []
            # We need d(EF)/d(layer4 output), not gradients through the whole video backbone.
            # Freeze parameter autograd temporarily to keep laptop memory bounded.
            flags = [p.requires_grad for p in model.parameters()]
            for p in model.parameters():
                p.requires_grad_(False)
            def capture(mod, inputs, output):
                activation = output.detach().requires_grad_(True)
                captured.append(activation)
                return activation
            hook = model.backbone.layer4.register_forward_hook(capture)
            try:
                with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
                    ef = model(x)['ef']
                    if abs(float(ef.detach())-prediction['outputs']['ef']) > .001:
                        raise ValueError('Explanation forward differs from displayed EF. No map shown.')
                    a = captured[0]
                    g = torch.autograd.grad(ef.sum(), a)[0]
                    raw = (g.mean((2,3,4),keepdim=True)*a).sum(1,keepdim=True).relu().detach()
            finally:
                hook.remove()
                captured.clear()
                for p, flag in zip(model.parameters(), flags):
                    p.requires_grad_(flag)
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite attribution. No map shown.')
            coarse = list(raw.shape[2:])
            peak = float(raw.max())
            if peak <= 0:
                return [], None, None, 'No positive EF Grad-CAM signal for this clip. This is method non-coverage, not proof that the model used no information.', None
            cam = torch.nn.functional.interpolate(raw, size=(32,112,112),mode='trilinear',align_corners=False)[0,0].cpu().numpy()
            cam /= cam.max()
            temporal = cam.mean((1,2))
            share = temporal / temporal.sum()
            overlays = []
            gallery = []
            for i, frame in enumerate(arr[idx]):
                gray = np.repeat(frame[:,:,None],3,axis=2)
                # Fixed color scale across the whole clip. Opacity fades to zero at zero relevance.
                color = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam[i]*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
                alpha = .55*cam[i,:,:,None]
                overlay = np.uint8(gray*(1-alpha)+color*alpha)
                overlays.append(overlay)
            for i in np.linspace(0,31,6).round().astype(int):
                gallery.append((overlays[i],f'Clip {i+1}/32 | source frame {int(idx[i])} (zero-based)'))
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8,2.8),tight_layout=True)
            ax = fig.subplots()
            ax.plot(np.arange(1,33),share*100,color='#315f54',linewidth=2)
            ax.set(xlabel='Sampled clip position (1-32)',ylabel='Relative influence (%)',title='How influence changes across the sampled clip')
            ax.grid(alpha=.15)
            ax.set_ylim(bottom=0)
            notice = ('Warm colors mark areas that contributed more toward a higher EF estimate. '
                      'They do not show disease or prove the model looked at the right anatomy. '
                      'Use the slider to compare the original frame with its overlay. '
                      f'The map is enlarged from a coarse {coarse[0]} x {coarse[1]} x {coarse[2]} grid; it cannot locate exact heart structures or phases.')
            frames = {'overlays':overlays,'raw':arr[idx], 'indices':idx.tolist()}
            return gallery, fig, overlays[0], notice, frames
    except (Exception, SystemExit) as exc:
        message = 'Explanation unavailable: '+str(exc)
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            message = 'Not enough GPU memory for gradients. Prediction is unchanged; choose CPU and analyze again to explain there.'
        return [], None, None, message, None


def select_frame(position, frames):
    if not frames:
        return None, None, 'Generate the influence view first.'
    i = max(0,min(31,int(position)-1))
    return frames['raw'][i], frames['overlays'][i], f"Clip position {i+1}/32 | source frame {frames['indices'][i]} (zero-based). No cardiac phase label is inferred."




CSS = '\n.gradio-container{max-width:1320px!important;width:100%!important;margin:auto!important;background:#fff!important;color:#172c4d;font-family:Arial,sans-serif!important;--body-text-color:#243c5b;--body-text-color-subdued:#61738e;--border-color-primary:#dde5f0;--block-background-fill:white;--input-background-fill:#f5f8fd;--button-primary-background-fill:#2c63e9;--button-primary-text-color:white;--button-primary-border-color:#2c63e9;--button-secondary-background-fill:#edf3ff;--button-secondary-text-color:#234b91;--block-radius:6px;--input-radius:5px;--button-large-radius:5px;--button-small-radius:5px}\n.gradio-container .main{padding:24px clamp(12px,3.2vw,42px)!important;box-sizing:border-box}.gradio-container *{min-width:0}footer{display:none!important}#brand{border-bottom:1px solid #dde5f0;padding:0 0 16px;margin-bottom:4px}#brand h1{font-size:27px;letter-spacing:-1px;line-height:1.2;font-weight:700;color:#152c50;margin:0}#brand p{font-size:12px;color:#71839e;margin:5px 0 0}\nbutton[role=tab]{min-height:46px!important;font-size:13px!important;font-weight:600!important;color:#667896!important;padding:12px 16px!important}button[role=tab][aria-selected=true]{color:#285de5!important;border-color:#285de5!important}#analyze-intro{padding:15px 0 18px}#analyze-intro h2{font-size:31px;font-weight:500;letter-spacing:-1px;line-height:1.25;margin:7px 0 9px;color:#142b4e}#analyze-intro p{font-size:14px;color:#687c97;line-height:1.6;margin:0}\n#video-panel{background:#14243d!important;border:0!important;border-radius:5px;padding:22px!important;gap:12px!important}#video-panel h3{color:#f1f5fc!important;font-size:17px!important;font-weight:600!important;margin:0!important}#video-panel>div.prose p,#video-panel .prose p{color:#a9b9d2!important;font-size:11px!important}#input-frame{background:#050a12!important;border:1px solid #30445f!important;border-radius:2px!important}#input-frame img{object-fit:contain!important}#video-panel .block:not(#input-frame){border:0!important;background:transparent!important}#video-panel label,#video-panel .label,#video-panel .label span{color:#c5d6ee!important}#video-panel label span,#video-panel .wrap span{color:#c8d5e9!important}#input-frame label,#input-frame .label,#input-frame .label span{color:#4d6485!important}#video-panel .prose p,#sample-status p{color:#bccbe0!important}#sample-status{font-size:12px!important;line-height:1.6!important}#refresh-samples{background:#243b5b!important;color:#d2e2ff!important;border:1px solid #3a5274!important;font-size:12px!important;min-height:40px!important}#analyze-button{background:#2c63e9!important;color:#fff!important;border:0!important;min-height:46px!important;font-size:14px!important;border-radius:5px!important}\n#result{padding:0!important;border:0!important;background:#fff!important}.result-card{padding:12px 16px 6px 24px}.result-identity{font-size:11px;letter-spacing:1px;font-weight:600;text-transform:uppercase;color:#4b6a9a;margin-bottom:21px}.eyebrow{font-size:11px;color:#6b7d98;letter-spacing:.4px}.ef-number{font-size:88px;line-height:1.08;letter-spacing:-4px;margin:6px 0 4px;color:#172e54}.ef-number span{font-size:34px;letter-spacing:-1px;margin-left:5px}.group-title{font-size:28px;font-weight:500;color:#2855a9;margin:0 0 6px}.group-summary{font-size:12px;line-height:1.65;color:#6c7e99;margin:0 0 20px}.ef-scale{height:6px;background:linear-gradient(to right,#c5cdd9 0%,#c5cdd9 40%,#adc2e3 40%,#adc2e3 50%,#6c96f7 50%);border-radius:4px;position:relative;margin:18px 0 8px}.ef-scale i{position:absolute;top:-5px;width:3px;height:16px;background:#24466e;border:1px solid white;transform:translateX(-50%)}.group-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.group-cell{display:flex;flex-direction:column;padding:7px 6px;gap:5px;color:#6c7e96;font-size:10px;line-height:1.45}.group-cell strong{font-size:12px;color:#3d557c}.group-cell.selected{background:#edf3ff;outline:1px solid #cadbff;border-radius:4px}.group-estimate{display:grid;grid-template-columns:125px minmax(0,1fr);gap:20px;padding:19px 0;margin:22px 0 0;border-top:1px solid #dce5f2;border-bottom:1px solid #dce5f2}.estimate-title{display:block;font-size:10px;font-weight:600;letter-spacing:.7px;color:#526e9d}.estimate-pct{font-size:45px;font-weight:600;line-height:1.2;letter-spacing:-1.5px;color:#285de5;margin:3px 0}.estimate-pct span{font-size:20px;letter-spacing:0;margin-left:3px}.estimate-kind{font-size:9px;letter-spacing:.7px;color:#6480ac;font-weight:600}.group-estimate strong{display:block;font-size:13px;line-height:1.5;color:#304a73;margin-top:2px}.group-estimate p{font-size:12px;line-height:1.65;color:#61738e;margin:7px 0 0}.group-estimate .est-note{color:#82653b}.estimate-unavailable,.separate-model{background:#f3f6fa;padding:13px 15px;color:#596e8c;font-size:12px;line-height:1.65;margin:18px 0}.estimate-unavailable strong{color:#355278}.estimate-unavailable p{margin:5px 0 0}.separate-model{margin:0 0 15px}.volume-row{display:flex;gap:44px;padding-top:17px}.volume-row div{display:flex;flex-direction:column;gap:6px}.volume-row span{font-size:11px;color:#6b7c96}.volume-row strong{font-size:23px;font-weight:500;color:#334d75}.volume-row small{font-size:13px}.ef-definition{font-size:11px;line-height:1.7;color:#6a7c96;margin:15px 0 0}.result-empty{min-height:430px;padding:35px 28px;display:flex;flex-direction:column;justify-content:center}.result-empty h2{font-size:32px;font-weight:500;letter-spacing:-1px;line-height:1.25;color:#213a62}.result-empty p{font-size:14px;line-height:1.8;color:#6c7f9b;max-width:390px}.empty-mark{font-size:62px;color:#dde5f3;margin-top:20px}.empty-mark span{font-size:25px}\n#reference-section{margin-top:22px;border-top:1px solid #dce5f2;padding-top:18px}#reference-section h2{font-size:22px;color:#20395f;font-weight:500}#comparison-view{font-size:13px;line-height:1.75;color:#4b6384}#comparison-view table{width:100%;font-size:13px}#comparison-view h3{font-size:20px;font-weight:500;color:#294b79}#compare-button{min-height:42px!important}#status{margin-top:12px}#technical{margin-top:16px}#runtime-settings{margin-top:12px}#runtime-settings .prose{font-size:12px;line-height:1.75}.section-intro h2{font-size:27px;letter-spacing:-.7px;font-weight:500;margin:22px 0 10px;color:#20395f}.section-intro p{font-size:13px;line-height:1.8;color:#637794}.frame-inspect img{width:100%!important;height:100%!important;object-fit:contain!important}#clip-gallery .grid-container{grid-template-columns:repeat(3,minmax(0,1fr))!important;grid-template-rows:repeat(2,160px)!important}#clip-gallery .thumbnail-lg{height:160px!important;min-height:0!important;aspect-ratio:auto!important}.scores-table{overflow-x:auto}.scores-table table{width:100%;border-collapse:collapse;font-size:14px}.scores-table th{text-align:left;font-size:11px;font-weight:500;color:#6e7e95;padding:15px 14px;border-bottom:1px solid #dde5ef}.scores-table td{padding:18px 14px;border-bottom:1px solid #e6ebf3;color:#324e78}.scores-table td:not(:first-child){font-size:20px;font-variant-numeric:tabular-nums}.scores-table th,.scores-table td{border-left:0!important;border-right:0!important;border-top:0!important}.scores-table small{display:block;font-size:10px;color:#476ca8;margin-top:4px}.scores-table .main-model{background:#edf3ff}.scope-note{padding:14px 18px;background:#edf3fd;color:#55729d;font-size:12px;line-height:1.7;border-radius:4px;margin:12px 0 20px}#influence-identity{font-size:12px;color:#4d6b9a;padding:10px 0}.gradio-container pre{white-space:pre-wrap!important;overflow-wrap:anywhere!important}.gradio-container .json-holder{overflow:auto;max-width:100%}.gradio-container .prose{overflow-wrap:anywhere}\n@media(max-width:900px){.gradio-container .main{padding:20px!important}.result-card{padding-left:12px}.ef-number{font-size:74px}.group-estimate{grid-template-columns:105px minmax(0,1fr);gap:12px}}\n@media(max-width:650px){.scores-table table,.scores-table tbody{display:block!important}.scores-table thead{display:none!important}.scores-table tr{display:block!important;padding:13px 12px!important;border-bottom:1px solid #dde5ef!important}.scores-table td{display:block!important;width:auto!important;border:0!important;padding:5px 0!important;font-size:14px!important;white-space:normal!important}.scores-table td:first-child{font-size:15px!important;font-weight:600!important;margin-bottom:8px!important}.scores-table td[data-label]{display:flex!important;justify-content:space-between!important;gap:16px!important;white-space:nowrap!important}.scores-table td[data-label]:before{content:attr(data-label);font-size:12px;color:#6a7d98}.gradio-container .main{padding:12px!important}#analyze-intro h2{font-size:26px}button[role=tab]{padding:10px!important;font-size:12px!important}#video-panel{padding:17px!important}.result-card{padding:20px 4px}.group-estimate{grid-template-columns:1fr;gap:9px}.estimate-pct{font-size:43px}.result-empty{min-height:260px;padding:20px 8px}.scores-table th,.scores-table td{padding:12px 7px}.scores-table td:not(:first-child){font-size:16px}.scores-table{font-size:12px}.group-cell strong{font-size:11px}#clip-gallery .grid-container{grid-template-columns:repeat(2,minmax(0,1fr))!important;grid-template-rows:repeat(3,145px)!important}#clip-gallery .thumbnail-lg{height:145px!important}#clip-gallery{height:490px!important}.volume-row{gap:28px}}\n\n#brand{text-align:center!important;padding:9px 0 22px!important}#brand h1{font-family:"Segoe UI Variable Display","Segoe UI",Arial,sans-serif!important;font-size:40px!important;line-height:1.15!important;font-weight:650!important;letter-spacing:-1.5px!important;color:#152c50!important}#brand p{font-family:"Segoe UI",Arial,sans-serif!important;font-size:13px!important;letter-spacing:.35px!important;color:#697f9e!important;margin-top:9px!important}\n'

DETAILS_TEXT = 'Temporary files: visualizations are periodically removed while the app runs, not securely erased. Max 4096 decoded frames.\n\n**What the numbers mean**\n\nEF is the estimated fraction of blood ejected (%). EDV and ESV are separately predicted end-diastolic and end-systolic volumes (mL). They need not agree exactly with the EF head. Their implied EF is recorded below, not used to repair the prediction.\n\nGroups use <40, 40 to <50 and >=50 EF cutoffs. They are not learned class outputs or calibrated probabilities. For the verified main baseline, a frozen estimator may show an experimental estimated chance that the predicted EF group is correct, only inside its fitted domain. It uses reused validation data, is not validated on new scans, and says nothing about the exact EF value. It is not a calibrated class probability. If its saved model, pipeline, source metrics or fitted domain checks fail, no percentage is shown.\n\nThe optional residual-error predictor uses a **different R3D mean checkpoint** and its paired uncertainty head. Its nominal range belongs only to that predictor. It uses selection-dependent validation calibration, not an independent clinical guarantee. A range crossing a group boundary does not give group probabilities. No training or recalibration runs here.\n\nOnly one centered 32-frame, stride-2 clip is analyzed at 112 x 112. Short videos wrap around. No whole-video, quality or view validation is implied. The separate four-class experiment is preserved outside this app.'
SCORE_INTRO = '## Model results\nThese are saved evaluation results, not predictions on the selected video. Lower EF error is better. All scores below use the official TEST set of 1,277 videos, which was used during development.'
SCORE_HTML = '<div class="scores-table"><table><thead><tr><th>Regression model</th><th>MAE · EF points</th><th>RMSE · EF points</th><th>R²</th></tr></thead><tbody><tr class="main-model"><td>R3D-18 baseline<small>Main app model</small></td><td data-label="MAE · EF points">5.379</td><td data-label="RMSE · EF points">7.345</td><td data-label="R²">0.639</td></tr><tr><td>Video Swin-T</td><td data-label="MAE · EF points">5.385</td><td data-label="RMSE · EF points">7.224</td><td data-label="R²">0.651</td></tr><tr><td>Frame ResNet-18 + mean pooling</td><td data-label="MAE · EF points">6.222</td><td data-label="RMSE · EF points">8.652</td><td data-label="R²">0.499</td></tr><tr><td>R3D + label-distribution smoothing</td><td data-label="MAE · EF points">5.962</td><td data-label="RMSE · EF points">7.690</td><td data-label="R²">0.604</td></tr><tr><td>R3D + class-balanced DRW</td><td data-label="MAE · EF points">5.662</td><td data-label="RMSE · EF points">7.742</td><td data-label="R²">0.599</td></tr></tbody></table></div>'
SCORE_MEANINGS = 'MAE is the average absolute difference between predicted and reference EF, in percentage points. It is a group average, not an error guarantee for your video. RMSE gives more weight to large errors. R2 measures explained variation, not a confidence percentage. The reported R2 is the coefficient of determination, not squared Pearson correlation. R3D and Swin have similar recorded MAE; an exploratory paired interval crossed zero, so there is no clear winner or proof of equivalence.'
VALIDATION_TEXT = 'On the 1,288 saved validation predictions, the main R3D baseline had MAE **5.467 EF points**, and **90.1%** of predictions were within **11.59 EF points** of the recorded EF. This is a fixed summary of past errors: the same 11.59 applies to every video. The validation data was also used during development, so this describes development-time performance - not a per-video confidence, not a prediction interval for your video, and not a guarantee for new videos. The Test comparison tab remains the per-sample check against the recorded EF.\n\nWhen the saved fit file group_correctness_estimate.json is present, the main result also shows an experimental estimated chance that the predicted group is correct. A small logistic estimator uses the predicted group and the distance of the estimated EF to the nearest group boundary; it was fitted on 645 validation rows and checked on the other 643, both reused during development. It appears only for the verified main checkpoint and only inside the fitted range; outside that range it is withheld. It is not validated on new scans, it says nothing about the exact EF value, and it is not medical advice or a referral rule.'
SOURCES_TEXT = 'Regression aggregates: results_recheck and AUDIT_SUMMARY.txt at reviewed source commit 68d2bc8. Baseline historical three-group F1 is 0.632867 in results/test_eval_baseline/test_metrics.json; the later evaluator reports 0.636675 in results/eval_runs/r3d18_baseline/metrics.json. They are different inference records, not interchangeable scores. Four-class numbers come from the separately supplied frozen epoch-18 report, prediction arrays and integrity manifest reviewed on 4 November 2026. The newest local work is not claimed to be present in the public commit. This interface does not re-run evaluation or select models.'
RESEARCH_INTRO = '## Additional model experiments\nThis residual-error research predictor is **separate from the main R3D baseline**: its own R3D mean checkpoint (`checkpoints/r3d18_mcdropout_best.pt`), a paired residual-uncertainty head (`checkpoints/r3d18_residual_uncertainty_best.pt`) and validation-fitted calibration. Its EF estimate and prediction range belong only to this experiment, never to the main result. Analyze a TEST sample first; the experiment runs on the same selected video.'
UNCERTAINTY_TEXT = 'Historical saved results: MC dropout Spearman correlation with error was about -0.04, so it did not reliably rank errors. Heteroscedastic sigma was constant and could not rank errors. The residual-error head showed a modest correlation of about 0.23. Historical nominal 90% ranges covered about 90.8%, 91.5% and 90.7% of TEST videos respectively. These are development-time results, not clinical guarantees; the historical aggregates do not fully identify their generating checkpoints.\n\nThe optional app residual predictor has its own paired mean checkpoint, source and calibration checks. Its range does not belong to the main baseline. No clinical low-confidence threshold was validated.'
FOURCLASS_TEXT = "The separate R3D four-class model adds learned class outputs: EF <40, 40 to <50, 50 to <70, and >=70. It is not the three-group predictor used by this app. Clean FP32 training completed 20 epochs; epoch 18 was selected using EF MAE on 643 validation videos. Temperature was fitted on 645 reserved validation videos before frozen TEST evaluation.\n\nTEST: EF MAE **5.9413 points**, RMSE **7.8470**, R2 **0.5880**, four-class macro F1 **0.4929**, accuracy **62.49%**. Accuracy is an aggregate, not a per-video confidence score. Bootstrap 95% intervals: EF MAE 5.6488 to 6.2174; macro F1 0.4595 to 0.5238.\n\nTemperature **0.901701** changed 15-bin ECE from 0.04552 to 0.03589, NLL from 0.85927 to 0.85779, and Brier score from 0.49884 to 0.49626. Changes are modest, not clinically validated or established as statistically significant. Temperature does not change EF or argmax/F1. These probabilities cannot be attached to another model's EF.\n\nThree versus four classes is not a matched comparison: definitions and training differ. We retain both results without claiming three classes are inherently superior."
PROJECT_INTRO = '## Inside Echo-EF\nExplore each topic below. The main tool estimates EF and volumes from a local TEST video, then derives one of three project groups. The influence view shows positive model attribution.'
PIPELINE_TEXT = '1. Select an available original video from TEST rows in the configured local FileList.csv. TRAIN and VAL are excluded. Read its frames, convert to grayscale and resize to 112 x 112.\n2. Choose one centered 32-frame clip with stride 2; short clips wrap around.\n3. Apply model-specific normalization and deterministic FP32 inference with TF32 off.\n4. Predict EF, EDV and ESV separately. Derive Reduced (<40%), Mildly reduced (40% to below 50%), or Normal (>=50%) from EF.\n5. On request, generate positive EF Grad-CAM from the same predictor, checkpoint and sampled input. The native 4 x 7 x 7 grid is enlarged for viewing. No heart segmentation or precise cardiac-phase detection is included.'
DATA_TEXT = "EchoNet-Dynamic: TRAIN 7,465, VAL 1,288, TEST 1,277 videos. Access requires the dataset owner's registration and agreement. Do not publish source videos, identifiers or derived frames. R3D and Video Swin use official torchvision Kinetics pretraining. The frame comparison uses ImageNet-pretrained ResNet-18 with temporal mean pooling, not EfficientNet-B3 + LSTM. W&B is used for experiment tracking; a claimed separately hosted weight source remains unverified. Local development used a 6 GB RTX 3060 Laptop GPU. Do not start training merely to run the existing checkpoints."
COMPLETED_TEXT = 'Completed experiments include regression models, uncertainty methods, explanation checks, a frozen ordinal-head experiment closed as inconclusive, and the separate learned four-class R3D baseline with reviewed calibration. A completed experiment can have a weak or negative result.\n\nOn 20 exploratory validation clips, explanation masking supported sensitivity to model-used information, not anatomical validity. Positive Grad-CAM maps were available in 17/20 clips; empty maps count as method non-coverage. Matched ED/ES alignment showed no clear improvement over its random baseline. Do not use the older unmatched 35.7% versus 29.0% comparison as validated improvement.\n\nAge/sex fairness could not be assessed because the available metadata has no validated demographic fields. TEST and VAL were used during development, not pristine external validation. Physiological bounds do not establish safety. The app has no automatic view or quality validation.'
REMAINING_TEXT = "Completed separately: all five comparable R3D four-class augmentation arms and the matched regression-only Swin control/MixUp pair. MixUp had the lowest selection EF MAE among the R3D arms. The Swin pair also favored MixUp on selection EF MAE, but one reused development split and one seed do not establish general superiority. These experiments do not replace the main app model. EfficientNet-B3 + LSTM remains deferred. Calibration is verified only for the retained original four-class head; optional winner calibration is paused, not completed, and cannot transfer to another checkpoint.\n\nImplemented recipe differs from the proposal: 32 stride-2 frames instead of 16 uniformly sampled frames; Swin uses 112 rather than 224 resolution; head/backbone optimizer groups do not establish layerwise LR decay. The app's three EF-derived groups replace the proposed learned four-class main output; the separate four-class experiment is retained.\n\nFull proposal completion, final report/poster/slides, public deployment, ONNX/API/Docker and complete laptop app acceptance are not claimed. No experiments restart from this interface."
PRIVACY_TEXT = 'Use only de-identified local TEST echo clips. Predictions can be wrong even when within plausible ranges. Files and visualizations stay on the computer running the app; they may remain in temporary storage until periodic cleanup and are not securely erased. Share links are disabled. The app does not train or recalibrate. Do not expose the local server to untrusted networks.'

def build_app():
    import gradio as gr
    theme = gr.themes.Base(primary_hue='blue', neutral_hue='slate', font=['Arial','sans-serif'])
    if tuple(int(x) for x in gr.__version__.split('.')[:2]) != (6,31):
        raise SystemExit('Tested with gradio==6.31.0. Keep torch/CUDA unchanged.')
    with gr.Blocks(title='Echo-EF', analytics_enabled=False, delete_cache=(3600,3600)) as app:
        gr.Markdown('# Echo-EF\nEjection fraction from echo video.', elem_id='brand')
        prediction = gr.State({})
        frame_state = gr.State(None)
        upload = gr.State(None)
        model = gr.State(R3D)
        device = gr.State('Auto')
        with gr.Tabs(overflow_behavior='wrap'):
            with gr.Tab('Analyze', id='analyze'):
                gr.Markdown('## See the video. Understand the estimate.\nChoose a local TEST sample. See its sampled frame, estimated EF and derived group together.', elem_id='analyze-intro')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=290, elem_id='video-panel'):
                        gr.Markdown('### The selected video')
                        preview = gr.Image(label='Middle sampled model input frame · after analysis', interactive=False, height=310, elem_id='input-frame', buttons=['fullscreen'])
                        gr.Markdown('A sampled frame, not video playback or a quality check. It appears after analysis.')
                        selector = gr.Dropdown(choices=[], label='Available TEST videos', filterable=True, value=None)
                        with gr.Row():
                            refresh = gr.Button('Refresh samples', elem_id='refresh-samples')
                            button = gr.Button('Analyze video', variant='primary', elem_id='analyze-button')
                        sample_status = gr.Markdown('Loading local TEST samples...', elem_id='sample-status')
                        gr.Markdown('The recorded EF is not used by the model. No upload, training or download starts here.')
                    with gr.Column(scale=6, min_width=300):
                        result = gr.HTML(result_card(), elem_id='result')
                        status = gr.Textbox(label='Analysis status', value='Choose a TEST sample to begin.', interactive=False, lines=2, elem_id='status')
                with gr.Column(elem_id='reference-section'):
                    gr.Markdown('## A reference, not an input.\nFor a verified original TEST sample, compare with its recorded EF only after prediction. A group match does not establish exact EF or overall accuracy.')
                    compare_button = gr.Button('Verify TEST identity and compare', elem_id='compare-button')
                    comparison_view = gr.Markdown('Actual EF unavailable. Analyze a TEST sample first.', elem_id='comparison-view')
            with gr.Tab('Model influence', id='influence'):
                gr.Markdown('## What influenced the estimate?\nInspect the original model input beside its positive EF attribution. Warm colors mean stronger influence toward a higher EF estimate, not disease. Generate after analysis; this may take longer.', elem_classes=['section-intro'])
                influence_identity = gr.Markdown('No prediction available. Analyze a TEST sample first.', elem_id='influence-identity')
                explain = gr.Button('Generate influence view', variant='primary')
                explanation = gr.Markdown('Analyze a video first, then generate its influence view.')
                slider = gr.Slider(1,32,value=1,step=1,label='Inspect sampled frame')
                with gr.Row():
                    raw_image = gr.Image(label='Original model input frame', interactive=False, height=340, elem_classes=['frame-inspect'], buttons=['fullscreen'])
                    overlay = gr.Image(label='Positive EF attribution overlay', interactive=False, height=340, elem_classes=['frame-inspect'], buttons=['fullscreen'])
                frame_label = gr.Markdown('No influence view generated.')
                with gr.Accordion('Across the sampled clip · six frames & temporal influence', open=False):
                    gallery = gr.Gallery(label='Across the sampled clip', columns=3, rows=2, height=380, object_fit='contain', allow_preview=False, interactive=False, buttons=[], elem_id='clip-gallery')
                    curve = gr.Plot(label='Frame attribution')
            with gr.Tab('Model results', id='comparison'):
                gr.Markdown(SCORE_INTRO, elem_classes=['section-intro'])
                gr.HTML('<div class="scope-note"><strong>Historical TEST evaluation · 1,277 videos.</strong><br>Saved regression results, not a prediction on the selected video. TEST was used during development, not pristine external validation.</div>')
                gr.HTML(SCORE_HTML)
                with gr.Accordion('How to read these scores', open=False):
                    gr.Markdown(SCORE_MEANINGS)
                with gr.Accordion('Validation errors & experimental group estimate', open=False):
                    gr.Markdown(VALIDATION_TEXT)
                with gr.Accordion('Sources, checkpoints & version notes', open=False):
                    gr.Markdown(SOURCES_TEXT)
            with gr.Tab('Research', id='experiments'):
                gr.Markdown(RESEARCH_INTRO, elem_classes=['section-intro'])
                experiment_button = gr.Button('Run separate residual-error experiment')
                experiment_view = gr.Markdown('Analyze a TEST sample in Analyze, then run the separate residual-error experiment here.')
                with gr.Accordion('Uncertainty experiments · historical results & limits', open=False):
                    gr.Markdown(UNCERTAINTY_TEXT)
                with gr.Accordion('Original four-class R3D experiment & calibration', open=False):
                    gr.Markdown(FOURCLASS_TEXT)
                with gr.Accordion('Completed augmentation & matched Swin studies', open=False):
                    gr.Markdown('All five R3D four-class augmentation arms completed comparable training. MixUp had the lowest selection EF MAE among those arms. A separate matched regression-only Swin control/MixUp pair also completed; MixUp had lower selection EF MAE in that pair. These are separate studies with different model outputs, not one common four-class comparison. Selection data was reused during development; single-seed results do not establish general superiority. These results do not replace the original baseline used by the app. Optional winner calibration remains paused; no new TEST evaluation runs here.')
        def controls(enabled):
            return [gr.update(interactive=enabled) for _ in range(6)]
        busy = [selector,refresh,button,explain,experiment_button,compare_button]
        def identity(details):
            if 'outputs' not in details:
                return 'No prediction available. Analyze a TEST sample first.'
            return ('Active predictor: **'+details['model']+'**. Checkpoint SHA-256: `'+details['checkpoint_sha256']+'`. The influence view must match this prediction and input. Changing the sample clears this view.')
        def analyze(path,choice,dev):
            try:
                current, _ = available_test_samples()
                if path not in current.values():
                    raise ValueError('Selected video is no longer an available TEST sample. Refresh the list.')
                text,img,details,state = run_prediction(path,choice,dev)
            except Exception as exc:
                text,img,details,state = result_card(),None,{},str(exc)
            return text,img,state,details,[],None,None,None,('Generate the influence view when ready.' if 'outputs' in details else 'No prediction available. Fix the input before generating influence.'),None,'No influence view generated.','Analyze a TEST sample, then run the separate residual-error experiment in Research.',(test_comparison(path,details) if 'outputs' in details else 'Actual EF unavailable. Analyze a TEST sample first.'),identity(details)
        outputs = [result,preview,status,prediction,gallery,curve,raw_image,overlay,explanation,frame_state,frame_label,experiment_view,comparison_view,influence_identity]
        def clear():
            return result_card(),None,'Input changed. Analyze again.',{},[],None,None,None,'Analyze again before generating influence.',None,'No influence view generated.','Analyze a TEST sample, then run the separate residual-error experiment in Research.','Actual EF unavailable. Verify the current video after analysis.','No prediction available. Analyze a TEST sample first.'
        def choose_sample(name):
            path,note = select_test_sample(name)
            return (path,note,*clear())
        app.load(refresh_test_samples,None,[selector,upload,sample_status],api_name=False)
        refresh_event = refresh.click(refresh_test_samples,None,[selector,upload,sample_status],api_name=False)
        refresh_event.then(clear,None,outputs,api_name=False)
        selector.change(choose_sample,selector,[upload,sample_status,*outputs],api_name=False,queue=False)
        event = button.click(lambda:controls(False),None,busy,api_name=False,queue=False)
        event = event.then(analyze,[upload,model,device],outputs,api_name=False)
        event.then(lambda:controls(True),None,busy,api_name=False,queue=False)
        e = explain.click(lambda:controls(False),None,busy,api_name=False,queue=False)
        e = e.then(run_explanation,[upload,model,device,prediction],[gallery,curve,overlay,explanation,frame_state],api_name=False)
        e = e.then(select_frame,[slider,frame_state],[raw_image,overlay,frame_label],api_name=False)
        e.then(lambda:controls(True),None,busy,api_name=False,queue=False)
        u = experiment_button.click(lambda:controls(False),None,busy,api_name=False,queue=False)
        u = u.then(check_uncertainty,[upload,device,prediction],experiment_view,api_name=False)
        u.then(lambda:controls(True),None,busy,api_name=False,queue=False)
        t = compare_button.click(lambda:controls(False),None,busy,api_name=False,queue=False)
        t = t.then(test_comparison,[upload,prediction],comparison_view,api_name=False)
        t.then(lambda:controls(True),None,busy,api_name=False,queue=False)
        slider.change(select_frame,[slider,frame_state],[raw_image,overlay,frame_label],api_name=False,queue=False)
    return app, {'theme':theme,'css':CSS}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be in 1..65535')
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Run from echo-ef-project: python app/app.py')
    app,ui_args = build_app()
    print('Local Echo-EF app. No training, recalibration or public share. Stop with Ctrl+C.')
    app.queue(default_concurrency_limit=1,max_size=4).launch(server_name='127.0.0.1',server_port=args.port,
        share=False,inbrowser=True,show_error=False,max_file_size='256mb',**ui_args)


if __name__ == '__main__':
    main()
