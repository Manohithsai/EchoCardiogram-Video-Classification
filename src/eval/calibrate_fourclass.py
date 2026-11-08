"""Frozen four-class baseline: reserved-VAL temperature fit, then fixed TEST evaluation.
Run from project root: python -m src.eval.calibrate_fourclass
No training, model selection, clinical claim or overwrite of existing results.
Outputs stay LOCAL: aggregate JSON/plots plus logits/labels/IDs for reproducibility.
TEST and VAL have already been used during project development, not pristine holdouts.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL = 'c082196e471a0c5db23bc5abeeb57ecca7bbc323aa69ba4e9ed957feb6f19802'
EXPECTED_TRAINER_LF = '87c2509bc2584d1bc76a2a655888cbe86e5e1ea04bf94a99a0326387c0a01c85'
CLASS_NAMES = ['EF below40', 'EF40 to below50', 'EF50 to below70', 'EF70 or above']
VERSION = 'frozen-fourclass-calibration-v1'
BINS = 15
BOOTSTRAPS = 1000
LOG_T_BOUNDS = (-4., 4.)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            d.update(b)
    return d.hexdigest()


def equivalent_source(path, expected):
    b = Path(path).read_bytes()
    lf = b.replace(b'\r\n', b'\n')
    return expected in {hashlib.sha256(x).hexdigest() for x in (b, lf, lf.replace(b'\n', b'\r\n'))}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(obj, path):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    temp = Path(str(path)+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def validate_arrays(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 2 or z.shape[1] != 4 or not len(z) or y.shape != (len(z),):
        raise ValueError('Expected nonempty Nx4 logits and N labels.')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all() or (y < 0).any() or (y > 3).any():
        raise ValueError('Invalid finite logits or four-class labels.')
    return z, y.astype(np.int64)


def probabilities(logits, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite.')
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nll(logits, labels, temperature=1.):
    from scipy.special import logsumexp
    z, y = validate_arrays(logits, labels)
    z = z / temperature
    return float(np.mean(logsumexp(z, axis=1)-z[np.arange(len(y)), y]))


def fit_temperature(logits, labels):
    """One scalar, NLL only, fixed search bounds; TEST never enters this function."""
    from scipy.optimize import minimize_scalar
    z, y = validate_arrays(logits, labels)
    result = minimize_scalar(lambda logt: nll(z, y, math.exp(logt)), bounds=LOG_T_BOUNDS,
                             method='bounded', options={'xatol': 1e-7, 'maxiter': 200})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Temperature optimizer failed; no TEST evaluation.')
    baseline = nll(z, y)
    # Identity is a prespecified allowed candidate, so fit cannot worsen calibration NLL.
    t = math.exp(float(result.x)) if result.fun < baseline else 1.
    return {'temperature': t, 'calibration_n': len(y), 'objective': 'NLL on reserved VAL calibration half only',
            'nll_before': baseline, 'nll_after': nll(z, y, t),
            'log_temperature_bounds': list(LOG_T_BOUNDS), 'optimizer': 'scipy bounded minimize_scalar; identity candidate',
            'near_search_boundary': abs(math.log(t)-LOG_T_BOUNDS[0]) < .01 or abs(math.log(t)-LOG_T_BOUNDS[1]) < .01,
            'note': 'Temperature does not change argmax classes or EF regression. No F1 improvement implied.'}


def reliability(logits, labels, temperature=1.):
    z, y = validate_arrays(logits, labels)
    p = probabilities(z, temperature)
    confidence, pred = p.max(1), p.argmax(1)
    # [0,1/15), ... last bin includes1. Fixed bins, not selected on results.
    indices = np.minimum((confidence*BINS).astype(int), BINS-1)
    rows = []
    for b in range(BINS):
        mask = indices == b
        n = int(mask.sum())
        rows.append({'lower': b/BINS, 'upper': (b+1)/BINS, 'n': n,
                     'mean_confidence': float(confidence[mask].mean()) if n else None,
                     'accuracy': float((pred[mask] == y[mask]).mean()) if n else None})
    ece = sum(r['n']/len(y)*abs(r['accuracy']-r['mean_confidence']) for r in rows if r['n'])
    brier = np.mean(np.sum((p-np.eye(4)[y])**2, axis=1))
    return {'n': len(y), 'ece_top_label_15_equal_width_bins': float(ece), 'nll': nll(z,y,temperature),
            'brier_multiclass_sum': float(brier), 'accuracy': float((pred==y).mean()), 'bins': rows}


def evaluate_arrays(data, temperature, bootstrap=True):
    from src.train.train_r3d_fourclass import metrics_fourclass
    z,y = validate_arrays(data['logits'],data['labels'])
    ef,true = np.asarray(data['ef'],float),np.asarray(data['true_ef'],float)
    if ef.shape != (len(y),) or true.shape != ef.shape or not np.isfinite(ef).all() or not np.isfinite(true).all():
        raise ValueError('Invalid EF predictions/targets.')
    out = metrics_fourclass(ef,true,z,y)
    corr = np.corrcoef(ef,true)[0,1] if np.std(ef)>0 and np.std(true)>0 else None
    out['pearson_r_ef'] = float(corr) if corr is not None and np.isfinite(corr) else None
    out['before_temperature'] = reliability(z,y)
    out['after_temperature'] = reliability(z,y,temperature)
    out['argmax_unchanged'] = bool(np.array_equal(z.argmax(1), probabilities(z,temperature).argmax(1)))
    if not out['argmax_unchanged']:
        raise RuntimeError('Unexpected class decision change; stop.')
    # Class-specific AUC can change after temperature since all four scores share the denominator.
    from sklearn.metrics import roc_auc_score
    p = probabilities(z,temperature)
    out['auc_ovr_after_temperature'] = [float(roc_auc_score(y==c,p[:,c])) if len(np.unique(y==c))==2 else None for c in range(4)]
    if bootstrap:
        from sklearn.metrics import f1_score
        rng=np.random.default_rng(20261104);mae=[];f1=[]
        pred=z.argmax(1)
        for _ in range(BOOTSTRAPS):
            ids=rng.integers(0,len(y),size=len(y))
            mae.append(float(np.abs(ef[ids]-true[ids]).mean()))
            f1.append(float(f1_score(y[ids],pred[ids],labels=[0,1,2,3],average='macro',zero_division=0)))
        out['bootstrap95_percentile']={'n_resamples':BOOTSTRAPS,'seed':20261104,'unit':'video; no patient-ID clustering available',
                                      'ef_mae':np.quantile(mae,[.025,.975]).tolist(),
                                      'macro_f1':np.quantile(f1,[.025,.975]).tolist()}
    return out


def verify_split(selection, calibration, val_n):
    if any(type(i) is not int for i in selection+calibration):
        raise ValueError('Split positions must be integers.')
    if not selection or not calibration or len(set(selection))!=len(selection) or len(set(calibration))!=len(calibration):
        raise ValueError('Empty/duplicate split positions.')
    if set(selection)&set(calibration) or sorted(selection+calibration)!=list(range(val_n)):
        raise ValueError('Selection/calibration overlap or incomplete partition.')


def predict_dataset(model, dataset, device, mean, std, label):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
    mu=torch.tensor(mean,device=device).view(1,3,1,1,1)
    sd=torch.tensor(std,device=device).view(1,3,1,1,1)
    arrays={k:[] for k in ('ef','true_ef','logits','labels','edv','esv','true_edv','true_esv')}
    names=[];t0=time.perf_counter()
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        for i,batch in enumerate(loader):
            out=model((batch['clip'].to(device)-mu)/sd)
            if not all(torch.isfinite(v).all() for v in out.values()):
                raise ValueError('Nonfinite frozen-model output; no skipped samples.')
            arrays['ef'].extend(out['ef'].cpu().tolist());arrays['true_ef'].extend(batch['ef'].tolist())
            arrays['logits'].extend(out['class_logits'].cpu().tolist());arrays['labels'].extend(batch['ef_class'].tolist())
            for k in ('edv','esv'):
                arrays[k].extend(out[k].cpu().tolist());arrays['true_'+k].extend(batch[k].tolist())
            names.extend(str(x) for x in batch['filename'])
            if i==0 or (i+1)%50==0:print(label,'videos',len(names),'/',len(dataset),flush=True)
    if len(names)!=len(dataset) or len(set(names))!=len(names):
        raise ValueError('Evaluation sample count or uniqueness mismatch.')
    result={k:np.asarray(v) for k,v in arrays.items()};result['file_names']=np.asarray(names,dtype=str)
    validate_arrays(result['logits'],result['labels'])
    if not all(np.isfinite(v).all() for k,v in result.items() if k!='file_names'):
        raise ValueError('Nonfinite predictions/targets.')
    print(label,'finished minutes',round((time.perf_counter()-t0)/60,2),flush=True)
    return result


def plot_reliability(metrics, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharex=True,sharey=True)
    for ax,key,title in zip(axes,('before_temperature','after_temperature'),('Before temperature','After temperature')):
        r=metrics[key];bins=[b for b in r['bins'] if b['n']]
        ax.plot([0,1],[0,1],linestyle='--',color='#aaa',linewidth=1)
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',color='#315f54')
        for b in bins:ax.annotate(str(b['n']),(b['mean_confidence'],b['accuracy']),xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted confidence',title=f"{title}\nECE {r['ece_top_label_15_equal_width_bins']:.4f} | NLL {r['nll']:.4f}")
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Observed class accuracy')
    fig.suptitle('Four-class frozen baseline | official TEST (previously used in development)',fontsize=11)
    fig.text(.5,.015,'15 fixed bins; empty bins omitted; point labels = videos per bin. No clinical calibration claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(path,dpi=160);plt.close(fig)


def load_npz(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def run(args):
    from src.data.echonet_dataset import EchoNetDataset,AugmentConfig
    from src.train import train_r3d_fourclass as trainer
    from torch.utils.data import Subset
    import torchvision
    import pandas as pd
    if Path.cwd().resolve()!=ROOT:raise SystemExit('Run from the echo-ef-project folder.')
    if not equivalent_source(trainer.__file__,EXPECTED_TRAINER_LF):raise SystemExit('Trainer differs from verified frozen experiment; stop, do not overwrite it.')
    ckpt_path=ROOT/'results/fourclass_fp32_v3/none/best.pt'
    run_root=ckpt_path.parent.parent
    if sha(ckpt_path)!=EXPECTED_MODEL:raise SystemExit('Selected checkpoint differs from verified epoch18; no fallback or re-selection.')
    complete=read_json(ckpt_path.parent/'complete.json')
    if complete['best_sha256']!=EXPECTED_MODEL or complete['best_epoch']!=18 or complete['epochs_completed']!=20:raise SystemExit('Completion record mismatch.')
    state=torch.load(ckpt_path,map_location='cpu',weights_only=True);p=state['protocol']
    if p!=read_json(run_root/'protocol.json') or state['epoch']!=18:raise SystemExit('Checkpoint/protocol mismatch.')
    if p['version']!='fourclass-fp32-staged-v3' or p['execution_arms']!=['none'] or p['cutoffs']!=[40.,50.,70.]:raise SystemExit('Unsupported experiment.')
    for key,file in [('source_sha256',trainer.__file__),('model_source_sha256',ROOT/'src/models/r3d_multitask.py'),('dataset_source_sha256',ROOT/'src/data/echonet_dataset.py')]:
        if not equivalent_source(file,p[key]):raise SystemExit('Computational source identity mismatch: '+file.__str__())
    cfg=p['config'];dc=cfg['data']
    csv=ROOT/dc['file_list_csv']
    if sha(csv)!=p['file_list_sha256']:raise SystemExit('FileList changed since model training.')
    if (dc['num_frames'],dc['frame_stride'],dc['frame_size'])!=(32,2,112):raise SystemExit('Unsupported preprocessing.')
    mean,std=p['norm_mean'],p['norm_std']
    if mean!=list(trainer.R3D_MEAN) or std!=list(trainer.R3D_STD):raise SystemExit('Normalization mismatch.')
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; use explicit --device cpu if desired.')
    device=torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    trainer.deterministic_seed(42)
    common=dict(cache_dir=str(ROOT/dc['cache_dir']),file_list_csv=str(csv),num_frames=32,frame_stride=2,ef_thresholds=[40,50,70],augment=AugmentConfig(temporal=False),seed=42)
    val=EchoNetDataset(split='VAL',**common)
    selection,calibration=p['selection_positions'],p['calibration_positions']
    verify_split(selection,calibration,len(val))
    if len(val)!=1288 or len(selection)!=643 or len(calibration)!=645 or (selection,calibration)!=trainer.stratified_val_split(val.df,42):raise SystemExit('Reserved split identity/count differs.')
    df=pd.read_csv(csv)
    ids={s:set(Path(str(n)).stem for n in df.loc[df['Split'].str.upper()==s,'FileName']) for s in ('TRAIN','VAL','TEST')}
    if any(ids[a]&ids[b] for a,b in [('TRAIN','VAL'),('TRAIN','TEST'),('VAL','TEST')]):raise SystemExit('Video IDs overlap across official splits.')
    if len(ids['TEST'])!=1277:raise SystemExit('TEST count/uniqueness differs from official1277.')
    source_path=Path(__file__)
    fixed={'version':VERSION,'checkpoint_sha256':EXPECTED_MODEL,'checkpoint_epoch':18,'trainer_source_lf':EXPECTED_TRAINER_LF,
           'evaluation_source_sha256':sha(source_path),'file_list_sha256':sha(csv),'temperature_log_bounds':list(LOG_T_BOUNDS),
           'bins':BINS,'bootstrap_resamples':BOOTSTRAPS,'batch_size':4,'sampling':'centered32frames/stride2/grayscale112; no augmentation',
           'selection_positions':selection,'calibration_positions':calibration,'config':cfg,
           'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'numpy':np.__version__,'device':str(device)},
           'caveats':['VAL and official TEST previously used in development; not independent untouched validation.',
                      'Single selected no-augmentation arm only; four other arms and winner-Swin deferred.',
                      'Classification experiment separate from EF-derived three-group main app; not a3-vs4 comparison.',
                      'No clinical validation or deployment claim; temperature is not a classification/F1 repair.']}
    out=ROOT/args.output
    if out.exists():
        if not args.resume:raise SystemExit('Output exists; preserve it. --resume only for this exact evaluation protocol.')
        if read_json(out/'evaluation_protocol.json')!=fixed:raise SystemExit('Evaluation source/data/runtime differs; no mixed-protocol resume.')
    else:
        if args.resume:raise SystemExit('No evaluation folder to resume.')
        out.mkdir(parents=True)
        atomic_json(fixed,out/'evaluation_protocol.json')
    lock=out/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('Evaluation lock exists. Stop, verify process state; do not delete blindly.')
    os.close(fd)
    try:
        if (out/'COMPLETE.json').exists():
            manifest=read_json(out/'COMPLETE.json')
            if not all(sha(out/name)==digest for name,digest in manifest['files'].items()):raise RuntimeError('Completed evaluation artifact hash mismatch.')
            print('ALREADY COMPLETE: verified saved report; no repeat inference or TEST re-evaluation.',flush=True);return
        model=trainer.R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).to(device)
        model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
        del state
        print('FROZEN epoch18; no training. Reserved645 calibration, fixed1277TEST. FP32, batch4.',flush=True)
        print('TEST was used during development; this is not a pristine external validation.',flush=True)
        cal_path=out/'calibration_predictions.npz'
        if not cal_path.exists():
            atomic_npz(cal_path,**predict_dataset(model,Subset(val,calibration),device,mean,std,'CALIBRATION'))
            atomic_json({'sha256':sha(cal_path)},out/'calibration_predictions.sha256.json')
        if sha(cal_path)!=read_json(out/'calibration_predictions.sha256.json')['sha256']:raise RuntimeError('Saved calibration prediction hash mismatch.')
        cal=load_npz(cal_path)
        cal_names=[str(x) for x in val.df.iloc[calibration]['FileName']]
        if cal['file_names'].tolist()!=cal_names:raise RuntimeError('Saved calibration identity differs.')
        validate_arrays(cal['logits'],cal['labels'])
        cal_true=val.df.iloc[calibration]['EF'].to_numpy(dtype=np.float32)
        if len(cal['labels'])!=645 or not np.array_equal(cal['true_ef'],cal_true) or not np.array_equal(cal['labels'],trainer.class_targets(torch.tensor(cal_true)).numpy()):raise RuntimeError('Calibration targets/labels differ before fitting.')
        # Calibration artifact persists BEFORE any TEST forward and binds its inputs.
        fit_path=out/'temperature.json'
        if fit_path.exists():
            if sha(fit_path)!=read_json(out/'temperature.sha256.json')['sha256']:raise RuntimeError('Saved temperature artifact hash mismatch.')
            fit=read_json(fit_path)
            if fit['calibration_predictions_sha256']!=sha(cal_path) or fit['checkpoint_sha256']!=EXPECTED_MODEL:raise RuntimeError('Saved temperature input identity differs.')
        else:
            fit=fit_temperature(cal['logits'],cal['labels'])
            fit.update(checkpoint_sha256=EXPECTED_MODEL,calibration_predictions_sha256=sha(cal_path),evaluation_source_sha256=sha(__file__))
            atomic_json(fit,fit_path)
            atomic_json({'sha256':sha(fit_path)},out/'temperature.sha256.json')
        print('Temperature',fit['temperature'],'calibration NLL',fit['nll_before'],'->',fit['nll_after'],flush=True)
        test=EchoNetDataset(split='TEST',**common)
        test_path=out/'test_predictions.npz'
        if not test_path.exists():
            atomic_npz(test_path,**predict_dataset(model,test,device,mean,std,'TEST'))
            atomic_json({'sha256':sha(test_path)},out/'test_predictions.sha256.json')
        if sha(test_path)!=read_json(out/'test_predictions.sha256.json')['sha256']:raise RuntimeError('Saved TEST prediction hash mismatch.')
        data=load_npz(test_path)
        if data['file_names'].tolist()!=[str(x) for x in test.df['FileName']]:raise RuntimeError('Saved TEST identity differs.')
        if len(cal['labels'])!=645 or len(data['labels'])!=1277:raise RuntimeError('Prediction counts differ.')
        for predictions,frame in ((cal,val.df.iloc[calibration]),(data,test.df)):
            true_ef=frame['EF'].to_numpy(dtype=np.float32)
            labels=trainer.class_targets(torch.tensor(true_ef)).numpy()
            if not np.array_equal(predictions['true_ef'],true_ef) or not np.array_equal(predictions['labels'],labels):raise RuntimeError('Saved target/label identity mismatch.')
        metrics=evaluate_arrays(data,fit['temperature'])
        for volume in ('edv','esv'):
            err=data[volume]-data['true_'+volume]
            metrics[volume+'_mae_ml']=float(np.abs(err).mean())
        metadata_columns=[str(c) for c in df.columns if str(c).strip().lower() in ('sex','gender','age','patientage','patient_age')]
        fairness={'file_list_columns':df.columns.tolist(),'potential_demographic_columns':metadata_columns,
                  'status':'Not assessed: no validated sex/age metadata is available in this pipeline. No proxy inference or fairness guarantee.',
                  'note':'If legitimate demographic metadata exists elsewhere, a separately grounded audit is needed; aggregate or EF-group metrics are not demographic fairness.'}
        report={'checkpoint_sha256':EXPECTED_MODEL,'selected_epoch':18,'split':'official TEST, previously used during development',
                'temperature':fit,'class_order':CLASS_NAMES,'test':metrics,'fairness':fairness,'caveats':fixed['caveats'],
                'calibration_support':np.bincount(cal['labels'].astype(int),minlength=4).tolist(),
                'test_support':np.bincount(data['labels'].astype(int),minlength=4).tolist(),
                'calibration_evaluation':'Fit-set NLL is not held-out evidence; TEST metrics before/after are reported, not optimized.',
                'physiology_flags':{'ef_outside_0_100':int(((data['ef']<0)|(data['ef']>100)).sum()),
                                    'invalid_volumes':int(((data['edv']<=0)|(data['esv']<0)|(data['esv']>data['edv'])).sum()),
                                    'policy':'All finite cases retained in evaluation; never remove errors to inflate scores.'}}
        atomic_json(report,out/'report.json')
        plot_reliability(metrics,out/'reliability.png')
        files=['evaluation_protocol.json','calibration_predictions.npz','calibration_predictions.sha256.json','temperature.json','temperature.sha256.json','test_predictions.npz','test_predictions.sha256.json','report.json','reliability.png']
        atomic_json({'status':'complete','files':{name:sha(out/name) for name in files}},out/'COMPLETE.json')
        print(json.dumps({'TEST_n':metrics['n'],'ef_mae':metrics['ef_mae'],'macro_f1':metrics['macro_f1'],
                          'ece_before':metrics['before_temperature']['ece_top_label_15_equal_width_bins'],
                          'ece_after':metrics['after_temperature']['ece_top_label_15_equal_width_bins'],
                          'argmax_unchanged':metrics['argmax_unchanged']},indent=2),flush=True)
        print('COMPLETE:',out,'Send report.json terminal contents and reliability.png only; prediction NPZ files contain dataset IDs and stay local.',flush=True)
    except BaseException as exc:
        atomic_json({'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error':str(exc),
                     'resume':'Same command plus --resume; completed split artifacts reused, incomplete split rerun. No checkpoint mutation.'},out/'last_failure.json')
        raise
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--output',default='results/fourclass_calibration_v1')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args)


if __name__=='__main__':main()
