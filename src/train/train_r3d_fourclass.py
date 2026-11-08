"""Separate R3D four-class variant and TRAIN-only timing pilot.

Bounded single-arm FP32 training is explicit --train-none-fp32 or --train-arm-fp32 ARM
(geometric, mixup, cutmix or temporal; one arm per launch, same frozen protocol/initialization).
Legacy AMP five-arm training remains separate --train-five. Protocol acknowledgement required.
Existing models/ordinal remain untouched. base.yaml read only. TRAIN+selection VAL only;
calibration half reserved; TEST not evaluated. --pilot never saves weights. Classes: <40,40-<50,50-<70,>=70.
"""
import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.r3d_multitask import R3DMultiTask, ef_consistency_loss

CUTOFFS = (40., 50., 70.)
R3D_MEAN = (0.43216, 0.394666, 0.37645)
R3D_STD = (0.22803, 0.22145, 0.216989)
CLASS_LOSS_WEIGHT = 1.0  # declared design choice, not a tuned/optimal value


class R3DFourClass(R3DMultiTask):
    """Keep original regression head; new head sees the same pooled features."""
    def __init__(self, pretrained=False, dropout_p=0.3, backbone_dropout_p=0.15):
        super().__init__(pretrained, dropout_p, backbone_dropout_p)
        self.class_head = nn.Linear(self.head[0].in_features, 4)

    def forward(self, clip):
        x = self.backbone.stem(clip)
        x = self.drop1(self.backbone.layer1(x))
        x = self.drop2(self.backbone.layer2(x))
        x = self.drop3(self.backbone.layer3(x))
        x = self.drop4(self.backbone.layer4(x))
        features = self.dropout(torch.flatten(self.backbone.avgpool(x), 1))
        out = self.head(features)
        return {'ef': out[:, 0], 'edv': out[:, 1], 'esv': out[:, 2],
                'class_logits': self.class_head(features)}


def class_targets(ef):
    # >= boundaries agree with dataset ef_to_class, including exactly40/50/70.
    if not torch.isfinite(ef).all():
        raise ValueError('nonfinite EF labels')
    return (ef[:, None] >= ef.new_tensor(CUTOFFS)[None]).sum(1)


def soft_class_targets(ef):
    return F.one_hot(class_targets(ef), num_classes=4).float()


def mixed_targets(batch, permutation, retained):
    """For MixUp/CutMix: interpolate ALL regression and ORIGINAL soft class labels.
    Do not threshold the mixed EF to manufacture a hard class target.
    """
    out = {k: retained * batch[k] + (1-retained) * batch[k][permutation]
           for k in ('ef', 'edv', 'esv')}
    q = soft_class_targets(batch['ef'])
    out['class_soft'] = retained*q + (1-retained)*q[permutation]
    return out


def mixup_batch(clip, batch, permutation, retained):
    if not 0 <= retained <= 1:
        raise ValueError('retained fraction must be in [0,1]')
    return retained*clip+(1-retained)*clip[permutation], mixed_targets(batch, permutation, retained)


def cutmix_batch(clip, batch, permutation, box):
    """One spatial rectangle across ALL time/channels. Actual clipped area sets labels."""
    x1,y1,x2,y2 = box
    h,w=clip.shape[-2:]
    if not (0<=x1<=x2<=w and 0<=y1<=y2<=h):
        raise ValueError('CutMix rectangle out of bounds')
    out=clip.clone()
    out[..., y1:y2, x1:x2]=clip[permutation,...,y1:y2,x1:x2]
    retained=1-((x2-x1)*(y2-y1))/(h*w)
    return out,mixed_targets(batch,permutation,retained)


def joint_loss(out, batch, consistency_weight=0.1):
    q=batch.get('class_soft')
    if q is None:
        q=soft_class_targets(batch['ef'])
    if not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6):
        raise ValueError('class targets must be finite probability distributions')
    reg=sum(F.smooth_l1_loss(out[k],batch[k]) for k in ('ef','edv','esv'))
    if consistency_weight:
        reg=reg+consistency_weight*ef_consistency_loss(out['ef'],out['edv'],out['esv'])
    cls=-(q*F.log_softmax(out['class_logits'],dim=1)).sum(1).mean()
    return reg+CLASS_LOSS_WEIGHT*cls


def selftest():
    torch.manual_seed(42);torch.set_num_threads(min(4,torch.get_num_threads()))
    assert class_targets(torch.tensor([39.9,40.,49.9,50.,69.9,70.])).tolist()==[0,1,1,2,2,3]
    batch={'ef':torch.tensor([30.,80.]),'edv':torch.tensor([100.,120.]),'esv':torch.tensor([70.,24.])}
    x=torch.stack((torch.zeros(3,4,4,4),torch.ones(3,4,4,4)));perm=torch.tensor([1,0])
    xm,bm=mixup_batch(x,batch,perm,.25)
    assert torch.all(xm[0]==.75) and torch.allclose(bm['class_soft'][0],torch.tensor([.25,0.,0.,.75]))
    xc,bc=cutmix_batch(x,batch,perm,(0,0,2,4))
    assert xc[0].mean()==.5 and bc['ef'][0]==55. and torch.allclose(bc['class_soft'][0],torch.tensor([.5,0.,0.,.5]))
    m=R3DFourClass(pretrained=False).eval()
    # Random/small synthetic input only; no downloaded weights/data.
    clip=torch.randn(2,3,8,32,32)
    out=m(clip);assert out['class_logits'].shape==(2,4)
    old=R3DMultiTask(pretrained=False).eval()
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('class_head.')})
    reference=old(clip)
    assert all(torch.equal(out[k],reference[k]) for k in ('ef','edv','esv'))
    loss=joint_loss(out,bm);loss.backward()
    assert torch.isfinite(loss) and m.class_head.weight.grad.abs().max()>0
    assert torch.isfinite(m.class_head.weight.grad).all() and m.head[0].weight.grad is not None
    print('SELFTEST PASSED: exact four-class boundaries; MixUp soft targets; CutMix actual-area labels; joint regression/classification gradients. regression output equals original architecture with matching weights. Synthetic only.')


def pilot(args):
    import yaml
    from torch.utils.data import DataLoader
    from src.data.echonet_dataset import EchoNetDataset, AugmentConfig
    if not torch.cuda.is_available():
        raise SystemExit('STOPPED: CUDA unavailable. Pilot requires laptop GPU, no CPU runtime extrapolation.')
    if args.batch_size<1 or args.steps<1 or args.warmup<1:
        raise SystemExit('batch size, measured steps, warmup must be positive')
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dc=cfg['data'];tc=cfg['train']
    ds=EchoNetDataset(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='TRAIN',
                     num_frames=dc['num_frames'],frame_stride=dc['frame_stride'],ef_thresholds=list(CUTOFFS),
                     augment=AugmentConfig(temporal=False),seed=cfg['seed'])
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)
    if not len(loader):
        raise SystemExit('STOPPED: fewer TRAIN videos than one batch')
    device=torch.device('cuda')
    model=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).to(device)
    # Timing-only starting biases keep EDV away from the consistency denominator clamp.
    # Not a trained/pilot performance result, and not a frozen full-training initialization.
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.tensor([55.,120.,54.],device=device))
    initial=copy.deepcopy(model.state_dict())
    mean=torch.tensor(R3D_MEAN,device=device).view(1,3,1,1,1)
    std=torch.tensor(R3D_STD,device=device).view(1,3,1,1,1)
    amp=bool(tc['amp']) and not args.fp32;times={}
    print('TIMING ONLY: random weights with regression biases55/120/54 for numerical timing stability; TRAIN only; no VAL/TEST/model downloads/checkpoint writes.')
    print('GPU:',torch.cuda.get_device_name(),'torch',torch.__version__)
    print('TRAIN size',len(ds),'batch',args.batch_size,'clip',dc['num_frames'],dc['frame_size'],'warmup',args.warmup,'measured',args.steps,'AMP',amp)
    for mode in ('frozen','unfrozen','forward'):
        model.load_state_dict(initial)
        for p in model.parameters():p.requires_grad_(True)
        if mode=='frozen':
            for p in model.backbone.parameters():p.requires_grad_(False)
        model.train(mode!='forward')
        if mode=='frozen':model.backbone.eval() # do not update frozen BatchNorm buffers
        opt=torch.optim.AdamW([
            {'params':list(model.head.parameters())+list(model.class_head.parameters()),'lr':tc['lr_head']},
            {'params':model.backbone.parameters(),'lr':tc['lr_backbone']}],weight_decay=tc['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=2.**12)
        it=iter(loader);samples=[];torch.cuda.reset_peak_memory_stats()
        for step in range(args.warmup+args.steps):
            torch.cuda.synchronize();start=time.perf_counter()
            try:batch=next(it)
            except StopIteration:it=iter(loader);batch=next(it)
            batch={k:batch[k].to(device) for k in ('clip','ef','edv','esv')}
            clip=(batch['clip']-mean)/std
            opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(mode!='forward'),torch.autocast(device_type='cuda',enabled=amp):
                out=model(clip)
                loss=joint_loss(out,batch,cfg['model']['ef_consistency_weight'])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite pilot loss; no silent skip')
            if mode!='forward':
                old_scale=scaler.get_scale()
                scaler.scale(loss).backward();scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip_norm'])
                if not torch.isfinite(norm):raise RuntimeError('nonfinite pilot gradients; no silent skip')
                scaler.step(opt);scaler.update()
                if scaler.get_scale()<old_scale:raise RuntimeError('AMP skipped a pilot optimizer step; stop and inspect')
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if step>=args.warmup:samples.append(elapsed)
        sec=sum(samples)/len(samples);times[mode]=sec
        print(f'{mode}: mean seconds/batch {sec:.4f}; peak allocated MiB {torch.cuda.max_memory_allocated()/2**20:.1f}; data loading included')
        del opt,scaler
    batches=math.ceil(len(ds)/args.batch_size)
    print('Rough compute-only TRAIN epoch minutes frozen/unfrozen:',round(times['frozen']*batches/60,2),round(times['unfrozen']*batches/60,2))
    print('Forward cost is on TRAIN clips, not a validation metric. Epoch projections exclude VAL, checkpointing, augmentation cost, early stopping and variability.')
    print('Pilot complete. No saved weights/results and no full run. Agree run budget and protocol before training5arms.')


# ----------------------------------------------------------------------------- bounded five-arm experiment
ARMS = ('none', 'geometric', 'mixup', 'cutmix', 'temporal')
FOCAL_GAMMA = 2.0


def focal_soft_loss(logits, targets, gamma=FOCAL_GAMMA):
    logp = F.log_softmax(logits.float(), dim=1)
    return -(targets * (1-logp.exp()).pow(gamma) * logp).sum(1).mean()


def full_joint_loss(out, batch, consistency_weight):
    q=batch.get('class_soft', soft_class_targets(batch['ef']))
    if not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6):
        raise ValueError('invalid soft class targets')
    reg=sum(F.smooth_l1_loss(out[k],batch[k]) for k in ('ef','edv','esv'))
    if consistency_weight:
        reg=reg+consistency_weight*ef_consistency_loss(out['ef'],out['edv'],out['esv'])
    return reg+CLASS_LOSS_WEIGHT*focal_soft_loss(out['class_logits'],q)


def geometric_clip(clip, rng):
    """One transform per video, shared by all frames. No wraparound pixel roll."""
    from torchvision.transforms import functional as TF
    b,c,t,h,w=clip.shape
    videos=[]
    for x in clip:
        frames=x.permute(1,0,2,3)
        if rng.random()<.5:frames=TF.hflip(frames)
        # Rotation/crop/colour jitter are the geometric arm specified in proposal.
        frames=TF.affine(frames,angle=float(rng.uniform(-10,10)),translate=[0,0],
                         scale=float(rng.uniform(.9,1.1)),shear=[0.,0.],
                         interpolation=TF.InterpolationMode.BILINEAR,fill=0.)
        crop=int(round(min(h,w)*float(rng.uniform(.9,1.))))
        top=int(rng.integers(0,h-crop+1));left=int(rng.integers(0,w-crop+1))
        frames=TF.resized_crop(frames,top,left,crop,crop,[h,w],TF.InterpolationMode.BILINEAR,antialias=True)
        frames=TF.adjust_brightness(frames,float(rng.uniform(.9,1.1)))
        frames=TF.adjust_contrast(frames,float(rng.uniform(.9,1.1)))
        videos.append(frames.permute(1,0,2,3))
    return torch.stack(videos).clamp(0,1)


def augment_batch(clip,batch,arm,rng):
    if arm=='geometric':return geometric_clip(clip,rng),batch
    if arm not in ('mixup','cutmix'):return clip,batch
    # Generate on CPU with its own numpy stream; same sample order across arms.
    perm=torch.tensor(rng.permutation(len(clip)),device=clip.device)
    lam=float(rng.beta(.4,.4))
    if arm=='mixup':return mixup_batch(clip,batch,perm,lam)
    h,w=clip.shape[-2:];ratio=math.sqrt(1-lam)
    cw,ch=int(w*ratio),int(h*ratio)
    cx,cy=int(rng.integers(0,w)),int(rng.integers(0,h))
    box=(max(0,cx-cw//2),max(0,cy-ch//2),min(w,cx+(cw+1)//2),min(h,cy+(ch+1)//2))
    return cutmix_batch(clip,batch,perm,box)


def centered_indices(total_frames,num_frames=32,stride=2):
    import numpy as np
    if total_frames<1:raise ValueError('empty video')
    span=num_frames*stride
    if total_frames<=span:return (np.arange(num_frames)*stride)%total_frames
    return (total_frames-span)//2+np.arange(num_frames)*stride


def stratified_val_split(df,seed):
    """Within each actual EF group, fixed random half selection/half calibration.
    No metrics used. Tiny groups (<2) cannot support both and stop explicitly.
    """
    import numpy as np
    labels=class_targets(torch.tensor(df['EF'].to_numpy(),dtype=torch.float32)).numpy()
    selection=[];calibration=[]
    for c in range(4):
        ids=np.flatnonzero(labels==c)
        if len(ids)<2:raise ValueError(f'VAL class{c} has fewer than2; cannot split')
        ids=np.random.default_rng(seed+100*c).permutation(ids)
        n=len(ids)//2;selection.extend(ids[:n].tolist());calibration.extend(ids[n:].tolist())
    return sorted(selection),sorted(calibration)


def metrics_fourclass(ef,true,logits,labels):
    import numpy as np
    from sklearn.metrics import f1_score,confusion_matrix,cohen_kappa_score,roc_auc_score
    pred=np.asarray(ef);true=np.asarray(true);prob=torch.softmax(torch.as_tensor(logits),dim=1).numpy()
    labels=np.asarray(labels);cls=prob.argmax(1);err=pred-true
    denom=float(((true-true.mean())**2).sum())
    per_auc=[]
    for c in range(4):
        y=(labels==c).astype(int)
        per_auc.append(float(roc_auc_score(y,prob[:,c])) if len(np.unique(y))==2 else None)
    return {'ef_mae':float(np.abs(err).mean()),'ef_rmse':float(np.sqrt((err**2).mean())),
            'ef_r2':float(1-(err**2).sum()/denom) if denom else None,
            'macro_f1':float(f1_score(labels,cls,labels=[0,1,2,3],average='macro',zero_division=0)),
            'kappa':float(cohen_kappa_score(labels,cls,labels=[0,1,2,3])) if len(np.unique(np.r_[labels,cls]))>1 else None,
            'auc_ovr_per_class':per_auc,'confusion_matrix':confusion_matrix(labels,cls,labels=[0,1,2,3]).tolist(),
            'n':len(labels)}


def file_hash(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(value,path):
    import json,os
    from pathlib import Path
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temp,p)


def atomic_checkpoint(value,path):
    import os
    from pathlib import Path
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');torch.save(value,temp);os.replace(temp,p)


def deterministic_seed(seed):
    import random,numpy as np
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def experiment_protocol(cfg,selection,calibration,train_df,val_df,fp32=False):
    import platform,torchvision
    from pathlib import Path
    protocol={'version':'fourclass-fivearm-v2-scalereplay','seed':42,'arms':list(ARMS),'epoch_cap':20,
            'frozen_epochs':4,'patience':6,'batch_size':4,'workers':0,'cutoffs':list(CUTOFFS),
            'pretrained':'R3D_18_Weights.KINETICS400_V1','selection_metric':'unrounded selection-half EF MAE; exact tie earliest epoch',
            'numerics':'AMP bounded same-batch scale-halving replay, restore preforward RNG/buffers, no skipped batch;12halvings/floor1',
            'loss':'EF/EDV/ESV smooth-L1 + config consistency + 1.0 soft-target focal CE gamma2',
            'sampler':'inverse-frequency 4class TRAIN weighted sampling, replacement, len(TRAIN)draws per epoch; shared epoch seeds',
            'augmentations':{'none':'centered32frameclip; no augment','geometric':'clip-consistent flip p.5, rotation +/-10deg, scale.9-1.1,crop.9-1,brightness/contrast.9-1.1',
                             'mixup':'Beta(.4,.4), mixed continuous+soft class labels, centered clips',
                             'cutmix':'Beta(.4,.4), shared spatial rectangle allframes, actual-area soft+continuous labels',
                             'temporal':'existing dataset TRAIN random start and stride jitter;32frames'},
            'norm_mean':list(R3D_MEAN),'norm_std':list(R3D_STD),'config':cfg,
            'train_n':len(train_df),'val_n':len(val_df),'selection_positions':selection,'calibration_positions':calibration,
            'file_list_sha256':file_hash(cfg['data']['file_list_csv']),
            'source_sha256':file_hash(__file__),
            'model_source_sha256':file_hash(Path(__file__).parents[1]/'models/r3d_multitask.py'),
            'dataset_source_sha256':file_hash(Path(__file__).parents[1]/'data/echonet_dataset.py'),
            'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__)},
            'caveats':['one seed, exploratory ablation; no performance guarantee',
                       'VAL used earlier in project; new split prevents NEW calibration/selection reuse, not globally untouched',
                       'Existing32frames/112resolution differ from proposal16frames/Swin224; documented existing recipe',
                       'Calibration half is reserved, not evaluated by this training runner; TEST never evaluated here',
                       'Mixing EDV/ESV/EF may violate exact physical consistency; report this augmentation limitation']}
    if fp32:
        protocol.update(version='fourclass-fp32-staged-v3', numerics='FP32 forward/loss/backward; autocast/scaling/TF32 disabled; finite gradient checks and standard clipping; hard stop on failure',
                        execution_arms=list(ARMS), approval_scope='Staged FP32 arms; each arm launched separately after explicit user go-ahead; no-augmentation arm completed in results/fourclass_fp32_v3',
                        comparison_status='Incomplete five-arm comparison; retain each arm regardless of result')
    return protocol


def full_experiment(args):
    import yaml,json,traceback,os
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader,Subset,WeightedRandomSampler
    from src.data.echonet_dataset import EchoNetDataset,AugmentConfig
    fp32=bool(getattr(args,'train_none_fp32',False) or getattr(args,'train_arm_fp32',None))
    if getattr(args,'train_none_fp32',False):execution_arms=('none',)
    elif getattr(args,'train_arm_fp32',None):execution_arms=(args.train_arm_fp32,)
    else:execution_arms=ARMS
    if not args.acknowledge_protocol:
        raise SystemExit('STOPPED: read protocol; --acknowledge-protocol required for bounded training')
    if not torch.cuda.is_available():raise SystemExit('STOPPED: CUDA required, no silent CPU run')
    with open(args.config) as f:cfg=yaml.safe_load(f)
    if cfg['data']['num_frames']!=32 or cfg['data']['frame_stride']!=2 or cfg['data']['frame_size']!=112:
        raise SystemExit('STOPPED: bounded protocol requires32frames/stride2/112; do not silently change recipe')
    if not fp32 and not cfg['train']['amp']:raise SystemExit('STOPPED: timing/budget protocol assumed AMP enabled')
    deterministic_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');dc=cfg['data'];tc=cfg['train']
    class AblationDataset(EchoNetDataset):
        def _sample_clip_indices(self,total_frames):
            if self.augment.temporal:
                return super()._sample_clip_indices(total_frames)
            return centered_indices(total_frames,self.num_frames,self.frame_stride)
    train_ds=AblationDataset(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='TRAIN',
                           num_frames=32,frame_stride=2,ef_thresholds=list(CUTOFFS),augment=AugmentConfig(temporal=False),seed=42)
    val_ds=EchoNetDataset(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='VAL',
                         num_frames=32,frame_stride=2,ef_thresholds=list(CUTOFFS),augment=AugmentConfig(temporal=False),seed=42)
    sel,cal=stratified_val_split(val_ds.df,42)
    protocol=experiment_protocol(cfg,sel,cal,train_ds.df,val_ds.df,fp32=fp32)
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True)
    protocol_path=root/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise SystemExit('STOPPED: protocol/source/data/runtime differs; do not reuse this run namespace')
        if not args.resume:raise SystemExit('STOPPED: run exists; use --resume for same protocol')
    else:
        if any(root.iterdir()):raise SystemExit('STOPPED: output directory not empty without a protocol')
        if args.resume:raise SystemExit('STOPPED: no existing protocol to resume')
        atomic_json(protocol,protocol_path)
    # Exclusive process lock prevents accidental two simultaneous writers; crash leaves a clear lock.
    lock=root/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('STOPPED: RUNNING.lock exists; verify old process stopped before removing stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    active=None
    try:
        initial=root/'initial.pt'
        if not initial.exists():
            source=Path(args.initial_from)
            if not source.exists():raise RuntimeError('original saved initialization missing; no fresh random restart')
            if fp32 and file_hash(source)!='a40aab51c0633e6ba7c11d7e5f58417313a5f955953bd5b24192bd699eb08f6b':
                raise RuntimeError('original v1 initialization hash mismatch; stop, no replacement weights')
            old=torch.load(source,map_location='cpu',weights_only=True)
            oldp=old['protocol']
            for key in ('config','file_list_sha256','selection_positions','calibration_positions','cutoffs','sampler','loss'):
                if oldp[key]!=protocol[key]:raise RuntimeError('original initialization protocol mismatch: '+key)
            atomic_checkpoint({'model_state_dict':old['model_state_dict'],'protocol':protocol,
                               'original_initial_sha256':file_hash(source)},initial)
            print('Reusing EXACT original failed-attempt starting weights. Failed v1 retained.',flush=True)
        init=torch.load(initial,map_location='cpu',weights_only=True)
        if init['protocol']!=protocol:raise RuntimeError('initial checkpoint protocol mismatch')
        if fp32 and init.get('original_initial_sha256')!='a40aab51c0633e6ba7c11d7e5f58417313a5f955953bd5b24192bd699eb08f6b':
            raise RuntimeError('original initialization provenance mismatch')
        init_hash=file_hash(initial)
        labels=class_targets(torch.tensor(train_ds.df['EF'].to_numpy(),dtype=torch.float32));counts=torch.bincount(labels,minlength=4)
        if (counts==0).any():raise RuntimeError('TRAIN missing one4class group')
        weights=(1./counts.double())[labels]
        print('TRAIN4class counts:',counts.tolist(),'selection/calibration sizes:',len(sel),len(cal),flush=True)
        print(f'Staged FP32 arm(s) {list(execution_arms)};20epochcap4frozen,patience6; about7hours+overhead per arm; no automatic other arms.' if fp32 else 'Legacy AMP five-arm run.',flush=True)
        print('Inverse-frequency sampler + focal gamma2; no TEST/calibration evaluation.',flush=True)
        mean=torch.tensor(R3D_MEAN,device=device).view(1,3,1,1,1);std=torch.tensor(R3D_STD,device=device).view(1,3,1,1,1)
        val_loader=DataLoader(Subset(val_ds,sel),batch_size=4,shuffle=False,num_workers=0)
        for arm in execution_arms:
            active=root/arm;active.mkdir(exist_ok=True)
            if (active/'complete.json').exists():
                complete=json.loads((active/'complete.json').read_text())
                if file_hash(active/'best.pt')!=complete['best_sha256']:
                    raise RuntimeError('completed checkpoint hash mismatch')
                print('Already complete, not rerunning:',arm,flush=True);continue
            m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).to(device)
            m.load_state_dict(init['model_state_dict'])
            opt=torch.optim.AdamW([{'params':list(m.head.parameters())+list(m.class_head.parameters()),'lr':tc['lr_head']},
                                   {'params':m.backbone.parameters(),'lr':tc['lr_backbone']}],weight_decay=tc['weight_decay'])
            scaler=torch.amp.GradScaler('cuda',enabled=not fp32,init_scale=2.**12)
            start=0;best=float('inf');patience=0;history=[];recovery_events=[]
            if (active/'last.pt').exists() or (active/'best.pt').exists():
                # A crash between atomic best and last writes may leave a newer best.
                candidates=[torch.load(active/name,map_location='cpu',weights_only=True) for name in ('last.pt','best.pt') if (active/name).exists()]
                state=max(candidates,key=lambda s:s['epoch'])
                if state['initial_sha256']!=init_hash or state['protocol']!=protocol:raise RuntimeError('resume provenance mismatch')
                m.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state_dict']);scaler.load_state_dict(state['scaler_state_dict'])
                start=state['epoch'];best=state['best_mae'];patience=state['patience'];history=state['history']
                recovery_events=state.get('recovery_events',[])
                del candidates,state
            for epoch in range(start,20):
                if patience>=6:break
                # Per-epoch seed supports exact EPOCH-boundary resume; interrupted epoch reruns from last completed state.
                deterministic_seed(42+epoch);rng=np.random.default_rng(4242+epoch)
                train_ds.rng=np.random.default_rng(42000+epoch)
                train_ds.augment=AugmentConfig(temporal=(arm=='temporal'))
                sampler=WeightedRandomSampler(weights,num_samples=len(train_ds),replacement=True,generator=torch.Generator().manual_seed(42+epoch))
                loader=DataLoader(train_ds,batch_size=4,sampler=sampler,num_workers=0,drop_last=False)
                frozen=epoch<4
                for p in m.backbone.parameters():p.requires_grad_(not frozen)
                m.train()
                if frozen:m.backbone.eval()
                t0=time.perf_counter();loss_sum=0.;seen=0
                for batch in loader:
                    batch={k:batch[k].to(device) for k in ('clip','ef','edv','esv')}
                    clip,batch_aug=augment_batch(batch['clip'],batch,arm,rng)
                    clip=(clip-mean)/std
                    if fp32:
                        loss_val,_,_=safe_fp32_batch(m,opt,clip,batch_aug,cfg['model']['ef_consistency_weight'],tc['grad_clip_norm'])
                        retries=[]
                    else:
                        loss_val,retries=safe_scaled_batch(m,opt,scaler,clip,batch_aug,cfg['model']['ef_consistency_weight'],tc['grad_clip_norm'])
                    for event in retries:
                        recovery_events.append(dict(epoch=epoch+1,batch=seen//4,**event))
                    loss=torch.tensor(loss_val)
                    n=len(clip);loss_sum+=float(loss.detach())*n;seen+=n
                m.eval();ef=[];true=[];logits=[];classes=[]
                with torch.no_grad():
                    for batch in val_loader:
                        clip=(batch['clip'].to(device)-mean)/std
                        # fp32 validation fixed across arms; no augmentation/MC dropout.
                        out=m(clip)
                        if not all(torch.isfinite(v).all() for v in out.values()):raise RuntimeError('nonfinite validation output')
                        ef.extend(out['ef'].cpu().tolist());true.extend(batch['ef'].tolist())
                        logits.extend(out['class_logits'].cpu().tolist());classes.extend(class_targets(batch['ef']).tolist())
                met=metrics_fourclass(ef,true,logits,classes)
                row={'epoch':epoch+1,'train_loss':loss_sum/seen,'selection':met,'seconds':time.perf_counter()-t0}
                history.append(row);improved=met['ef_mae']<best
                if improved:best=met['ef_mae'];patience=0
                else:patience+=1
                payload={'arch':'r3d18_fourclass','model_state_dict':m.state_dict(),'epoch':epoch+1,'val_mae':met['ef_mae'],
                         'norm_mean':list(R3D_MEAN),'norm_std':list(R3D_STD),'protocol':protocol,'initial_sha256':init_hash,
                         'optimizer_state_dict':opt.state_dict(),'scaler_state_dict':scaler.state_dict(),
                         'best_mae':best,'patience':patience,'history':history,'recovery_events':recovery_events}
                if improved:atomic_checkpoint(payload,active/'best.pt')
                atomic_checkpoint(payload,active/'last.pt');atomic_json(history,active/'history.json')
                print(f'{arm} epoch{epoch+1}/20 selectionMAE {met["ef_mae"]:.4f} macroF1 {met["macro_f1"]:.4f} best {best:.4f} patience{patience}/6 minutes{row["seconds"]/60:.2f}',flush=True)
            best_state=torch.load(active/'best.pt',map_location='cpu',weights_only=True)
            atomic_json({'arm':arm,'best_epoch':best_state['epoch'],'best_selection_mae':best,
                         'selection_metrics':best_state['history'][best_state['epoch']-1]['selection'],
                         'best_sha256':file_hash(active/'best.pt'),'epochs_completed':len(history),'numerical_recoveries':recovery_events},active/'complete.json')
            del m,opt,scaler;torch.cuda.empty_cache()
        if fp32:
            summaries=[json.loads((root/a/'complete.json').read_text()) for a in ARMS if (root/a/'complete.json').exists()]
            pending=[a for a in ARMS if not (root/a/'complete.json').exists()]
            if pending:
                atomic_json({'arms':summaries,'comparison_complete':False,'pending_arms':pending,
                             'scope':'Staged FP32 arms; five-arm comparison incomplete; no augmentation winner selected',
                             'calibration':'reserved; not fit/evaluated here','test':'not read/evaluated'},root/'summary.json')
                print('STAGED FP32 COMPLETE:',[x['arm'] for x in summaries],'PENDING:',pending,'Review before next arm. No automatic next run.',flush=True)
                return
            winner=min(summaries,key=lambda x:(x['best_selection_mae'],ARMS.index(x['arm'])))
        else:
            summaries=[json.loads((root/a/'complete.json').read_text()) for a in execution_arms]
            winner=min(summaries,key=lambda x:(x['best_selection_mae'],ARMS.index(x['arm'])))
        atomic_json({'arms':summaries,'winner_by_selection_mae':winner['arm'],'tie_rule':'fixed arm order none/geometric/mixup/cutmix/temporal',
                     'calibration':'reserved; not fit/evaluated here','test':'not read/evaluated','scope':'R3D only; Swin/B3 not launched'},root/'summary.json')
        print('ALL5ARMS COMPLETE. Winner by fixed selection MAE:',winner['arm'],'Review before calibration/TEST/Swin. No automatic next training.',flush=True)
    except BaseException as e:
        # Record interrupted/nonfinite/OOM failures; do not mark an incomplete arm complete.
        atomic_json({'status':'interrupted' if isinstance(e,KeyboardInterrupt) else 'failed','arm':str(active.name) if active else None,
                     'error':str(e),'traceback':traceback.format_exc(),'resume':'epoch-boundary only; interrupted epoch reruns'},root/'last_failure.json')
        raise
    finally:
        lock.unlink(missing_ok=True)


def selftest_full():
    import pandas as pd,numpy as np,tempfile
    from pathlib import Path
    selftest()
    q=torch.tensor([[.25,0.,0.,.75],[0.,1.,0.,0.]])
    logits=torch.randn(2,4,requires_grad=True)
    assert torch.allclose(focal_soft_loss(logits,q,0),-(q*F.log_softmax(logits,1)).sum(1).mean())
    loss=focal_soft_loss(logits,q);loss.backward();assert torch.isfinite(logits.grad).all()
    df=pd.DataFrame({'EF':[30.]*5+[45.]*6+[60.]*7+[80.]*4})
    s,c=stratified_val_split(df,42);assert not set(s)&set(c) and sorted(s+c)==list(range(len(df)))
    assert (s,c)==stratified_val_split(df,42)
    assert centered_indices(80).tolist()==list(range(8,72,2))
    assert centered_indices(3,4,2).tolist()==[0,2,1,0]
    x=torch.rand(2,3,4,32,32);x[:,:,1:]=x[:,:,:1]
    batch={'ef':torch.tensor([30.,80.]),'edv':torch.tensor([100.,120.]),'esv':torch.tensor([70.,24.])}
    for arm in ARMS:
        out,b=augment_batch(x,batch,arm,np.random.default_rng(7))
        assert out.shape==x.shape and torch.isfinite(out).all()
        assert torch.equal(out[:,:,0],out[:,:,1]) # spatial/mixing transform same across time
        if 'class_soft' in b:assert torch.allclose(b['class_soft'].sum(1),torch.ones(2))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'state.pt';atomic_checkpoint({'state':torch.tensor([1.,2.]),'epoch':3},p)
        assert torch.equal(torch.load(p,weights_only=True)['state'],torch.tensor([1.,2.]))
        atomic_json({'complete':False},Path(d)/'test.json')
    met=metrics_fourclass([30.,45.,60.,80.],[30.,45.,60.,80.],torch.eye(4)*10,[0,1,2,3])
    assert met['macro_f1']==1. and met['ef_mae']==0. and met['kappa']==1.
    print('FULL SELFTEST PASSED: focal/soft-label gradients, disjoint reproducible stratified split, all5arm shape/time consistency, atomic checkpoint roundtrip. No real data/training.')


def diagnose_numerics(args):
    """Read saved initialization; no optimizer updates, output/checkpoint writes, or VAL reads."""
    import yaml
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from src.data.echonet_dataset import EchoNetDataset,AugmentConfig
    if not torch.cuda.is_available():raise SystemExit('CUDA required for the actual AMP diagnostic')
    initial=Path(args.initial_from)
    if not initial.exists():raise SystemExit('STOPPED: saved initial.pt missing; no reconstruction')
    state=torch.load(initial,map_location='cpu',weights_only=True)
    cfg=state['protocol']['config'];dc=cfg['data']
    if file_hash(dc['file_list_csv'])!=state['protocol']['file_list_sha256']:
        raise SystemExit('STOPPED: FileList differs from saved experiment')
    ds=EchoNetDataset(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='TRAIN',
                      num_frames=32,frame_stride=2,ef_thresholds=list(CUTOFFS),augment=AugmentConfig(temporal=False),seed=42)
    # Explicit centered none-arm indexing, same as planned full run.
    import types
    ds._sample_clip_indices=types.MethodType(lambda self,n:centered_indices(n,32,2),ds)
    labels=class_targets(torch.tensor(ds.df['EF'].to_numpy(),dtype=torch.float32));counts=torch.bincount(labels,minlength=4)
    sampler=WeightedRandomSampler((1./counts.double())[labels],num_samples=len(ds),replacement=True,
                                  generator=torch.Generator().manual_seed(42))
    loader=DataLoader(ds,batch_size=4,sampler=sampler,num_workers=0)
    deterministic_seed(42);m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).cuda()
    m.load_state_dict(state['model_state_dict']);m.train();m.backbone.eval()
    for p in m.backbone.parameters():p.requires_grad_(False)
    mean=torch.tensor(R3D_MEAN,device='cuda').view(1,3,1,1,1);std=torch.tensor(R3D_STD,device='cuda').view(1,3,1,1,1)
    print('READ-ONLY NUMERICAL PROBE: saved shared initialization, first3planned TRAIN batches, no optimizer steps/results writes.')
    print('Later failed training batch may have different weights; this probe diagnoses initialization only, not its exact failed state.')
    print('initial SHA256',file_hash(initial),'GPU',torch.cuda.get_device_name())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for i,batch in enumerate(loader):
        if i==3:break
        batch={k:batch[k].cuda() for k in ('clip','ef','edv','esv')};clip=(batch['clip']-mean)/std
        for mode,amp,scale in (('AMP4096',True,4096.),('AMP1',True,1.),('FP32',False,1.)):
            deterministic_seed(42+i);m.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=amp):
                out=m(clip);loss=full_joint_loss(out,batch,cfg['model']['ef_consistency_weight'])
            (loss*scale).backward()
            grads=[p.grad for p in m.parameters() if p.grad is not None]
            invalid=sum(int((~torch.isfinite(g)).sum()) for g in grads)
            maxg=max(float(g.abs().max()) for g in grads)
            print(f'batch{i} {mode}: loss={float(loss.detach()):.6g}; nonfinite_grad_elements={invalid}; max_scaled_abs_gradient={maxg:.6g}',flush=True)
            print('  EF/EDV/ESV min,max:',[(k,float(out[k].min()),float(out[k].max())) for k in ('ef','edv','esv')],flush=True)
    print('Probe complete; saved initialization unchanged; no optimizer steps. Send full output; do not resume training yet.')


def gradient_diagnostics(model):
    """Count bad elements separately from norm-reduction overflow, before clipping."""
    bad=0;finite_sq=0.;max_abs=0.;bad_names=[]
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.detach();mask=torch.isfinite(g);n=int((~mask).sum());bad+=n
        if n:bad_names.append((name,n))
        if mask.any():
            finite=g[mask].double();finite_sq+=float((finite*finite).sum());max_abs=max(max_abs,float(finite.abs().max()))
    return {'nonfinite_elements':bad,'bad_parameter_counts':bad_names,
            'finite_elements_float64_norm':math.sqrt(finite_sq),'finite_max_abs':max_abs}


def capture_training_failure(args):
    """At most ONE frozen TRAIN epoch, exact saved init; private numerical diagnostic only."""
    import json,numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from src.data.echonet_dataset import EchoNetDataset,AugmentConfig
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    initial=Path(args.initial_from);root=initial.parent
    if not initial.exists():raise SystemExit('STOPPED: original initialization missing')
    state=torch.load(initial,map_location='cpu',weights_only=True);cfg=state['protocol']['config'];dc=cfg['data'];tc=cfg['train']
    if file_hash(dc['file_list_csv'])!=state['protocol']['file_list_sha256']:raise SystemExit('FileList changed')
    private=root/'numerical_probe'
    if private.exists():raise SystemExit('STOPPED: numerical_probe already exists; preserve previous diagnostic')
    private.mkdir()
    class Centered(EchoNetDataset):
        def _sample_clip_indices(self,n):return centered_indices(n,self.num_frames,self.frame_stride)
    ds=Centered(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='TRAIN',num_frames=32,frame_stride=2,
                ef_thresholds=list(CUTOFFS),augment=AugmentConfig(temporal=False),seed=42)
    labels=class_targets(torch.tensor(ds.df['EF'].to_numpy(),dtype=torch.float32));counts=torch.bincount(labels,minlength=4)
    sampler=WeightedRandomSampler((1./counts.double())[labels],num_samples=len(ds),replacement=True,generator=torch.Generator().manual_seed(42))
    loader=DataLoader(ds,batch_size=4,sampler=sampler,num_workers=0,drop_last=False)
    # Match original model construction and epoch-seeding order exactly.
    m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).cuda();m.load_state_dict(state['model_state_dict'])
    opt=torch.optim.AdamW([{'params':list(m.head.parameters())+list(m.class_head.parameters()),'lr':tc['lr_head']},
                           {'params':m.backbone.parameters(),'lr':tc['lr_backbone']}],weight_decay=tc['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=2.**12)
    deterministic_seed(42);ds.rng=np.random.default_rng(42000)
    for p in m.backbone.parameters():p.requires_grad_(False)
    m.train();m.backbone.eval()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mean=torch.tensor(R3D_MEAN,device='cuda').view(1,3,1,1,1);std=torch.tensor(R3D_STD,device='cuda').view(1,3,1,1,1)
    print('DIAGNOSTIC REPLAY: at most first frozen TRAIN epoch from original shared init; temporary model updates only; no VAL/TEST/arm-success checkpoint.',flush=True)
    print('Private failure state may contain video tensors/labels; KEEP numerical_probe LOCAL, OFF GITHUB.',flush=True)
    for step,batch in enumerate(loader):
        batch={k:batch[k].cuda() for k in ('clip','ef','edv','esv')};clip=(batch['clip']-mean)/std
        opt.zero_grad(set_to_none=True);cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        with torch.autocast(device_type='cuda',enabled=True):
            out=m(clip);loss=full_joint_loss(out,batch,cfg['model']['ef_consistency_weight'])
        scaler.scale(loss).backward();scaler.unscale_(opt)
        # Standard clipping remains unchanged. Count element-level failures BEFORE clipping alters gradients.
        diag=gradient_diagnostics(m)
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),tc['grad_clip_norm'])
        failed=not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm))
        if failed:
            report={'step_zero_based':step,'loss':str(float(loss.detach())),'scale':scaler.get_scale(),
                    'standard_norm':str(float(norm)),'before_clip':diag,
                    'output_ranges':{k:[str(float(out[k].detach().min())),str(float(out[k].detach().max()))] for k in ('ef','edv','esv')}}
            print('FAILURE CAPTURE',report,flush=True)
            # Original weights at the offending forward are unchanged since previous completed update.
            atomic_checkpoint({'model_state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict(),
                               'batch':batch,'cpu_rng':cpu_rng,'gpu_rng':gpu_rng,'step':step,
                               'initial_sha256':file_hash(initial)},private/'failure_state.pt')
            alternatives=[]
            for label,amp in (('AMP_unscaled',True),('FP32_unscaled',False)):
                opt.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
                with torch.autocast(device_type='cuda',enabled=amp):
                    alt=m(clip);alt_loss=full_joint_loss(alt,batch,cfg['model']['ef_consistency_weight'])
                alt_loss.backward();d=gradient_diagnostics(m)
                r={'mode':label,'loss':str(float(alt_loss.detach())),'gradients':d}
                alternatives.append(r);print('SAME-STATE COMPARISON',r,flush=True)
            # JSON can encode strings for invalid numerical diagnostics without inventing finite values.
            report['loss']=str(report['loss']);report['alternatives']=alternatives
            atomic_json(report,private/'failure_report.json')
            print('Stopped at first invalid step; no optimizer update for that batch. Original experiment untouched. Send terminal output.',flush=True)
            return
        scaler.step(opt);scaler.update()
        if step%200==0:print('Diagnostic finite completed steps:',step+1,flush=True)
    atomic_json({'status':'not reproduced in first frozen epoch','steps':step+1},private/'failure_report.json')
    print('NOT REPRODUCED in first frozen epoch. No further epoch attempted. Original experiment unchanged.',flush=True)


def safe_fp32_batch(model,opt,clip,batch,consistency_weight,grad_clip,do_step=True):
    """Same full-precision computation as saved-state gate, no fallback or skipped batch."""
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=clip.device.type,enabled=False):
        out=model(clip.float());loss=full_joint_loss(out,batch,consistency_weight)
    if not all(torch.isfinite(v).all() for v in out.values()) or not torch.isfinite(loss):
        raise RuntimeError('nonfinite FP32 forward/loss; no skip or precision fallback')
    loss.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('nonfinite FP32 gradients; no optimizer update')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
    if not torch.isfinite(norm):raise RuntimeError('FP32 gradient norm invalid; no optimizer update')
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('nonfinite clipped FP32 gradients; no optimizer update')
    if do_step:
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('nonfinite FP32 parameter after optimizer; no checkpoint saved for this epoch')
    return float(loss.detach()),float(norm),{k:v.detach() for k,v in out.items()}


def safe_scaled_batch(model,opt,scaler,clip,batch,consistency_weight,grad_clip,do_step=True):
    """Replay SAME pre-step weights/buffers/RNG on scale overflow; never skip data.
    Forward precision/loss/architecture unchanged. Max12 scale halvings, floor1.
    """
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    events=[]
    for retry in range(13):
        if retry:
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            with torch.no_grad():
                for k,v in model.named_buffers():v.copy_(buffers[k])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',enabled=True):
            out=model(clip);loss=full_joint_loss(out,batch,consistency_weight)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite forward loss; no skip or precision fallback')
        scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
        # Inspect all elements before standard clip can change them.
        bad=sum(int((~torch.isfinite(p.grad)).sum()) for p in model.parameters() if p.grad is not None)
        if bad:
            if scale<=1 or retry==12:raise RuntimeError('nonfinite gradients persist at bounded retry limit; no data skipped')
            new_scale=max(1.,scale/2)
            events.append({'retry':retry+1,'old_scale':scale,'new_scale':new_scale,'nonfinite_elements':bad})
            print('AMP SCALE RECOVERY same batch, same RNG/buffers, no optimizer update:',events[-1],flush=True)
            scaler.update(new_scale=new_scale)
            continue
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
        if not torch.isfinite(norm):raise RuntimeError('finite-gradient norm reduction overflow; stop, no silent alternate clipping')
        if do_step:
            scaler.step(opt);scaler.update()
        return float(loss.detach()),events
    raise RuntimeError('unreachable retry boundary')


def verify_saved_failure(args):
    """Use exact private failed state locally; no persisted optimizer update/results write."""
    from pathlib import Path
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    path=Path(args.initial_from).parent/'numerical_probe/failure_state.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    original=torch.load(args.initial_from,map_location='cpu',weights_only=True);cfg=original['protocol']['config']
    if saved['initial_sha256']!=file_hash(args.initial_from):raise SystemExit('STOPPED: original init hash mismatch')
    m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).cuda();m.load_state_dict(saved['model_state_dict'])
    for p in m.backbone.parameters():p.requires_grad_(False)
    m.train();m.backbone.eval();tc=cfg['train']
    opt=torch.optim.AdamW([{'params':list(m.head.parameters())+list(m.class_head.parameters()),'lr':tc['lr_head']},
                           {'params':m.backbone.parameters(),'lr':tc['lr_backbone']}],weight_decay=tc['weight_decay'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=4096.)
    batch={k:v.cuda() for k,v in saved['batch'].items()}
    mean=torch.tensor(R3D_MEAN,device='cuda').view(1,3,1,1,1);std=torch.tensor(R3D_STD,device='cuda').view(1,3,1,1,1)
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['gpu_rng'])
    loss,events=safe_scaled_batch(m,opt,scaler,(batch['clip']-mean)/std,batch,cfg['model']['ef_consistency_weight'],tc['grad_clip_norm'],do_step=False)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    assert events,'expected original scaled-overflow case to be reproduced; inspect before training'
    print('SAVED FAILURE RETRY VERIFIED: original step',saved['step'],'finite same-batch gradients after',len(events),'scale reductions; final scale',scaler.get_scale(),'loss',loss)
    print('No optimizer update, changed model state, or files written. Send output before new full run.')


def loss_breakdown(out,batch,consistency_weight,force_float=False):
    values={k:(v.float() if force_float else v) for k,v in out.items()}
    result={}
    for k in ('ef','edv','esv'):
        v=values[k]
        result[k+'_range']=[str(float(v.detach().min())),str(float(v.detach().max()))]
        result[k+'_finite']=bool(torch.isfinite(v).all())
        result[k+'_smooth_l1']=str(float(F.smooth_l1_loss(v,batch[k]).detach()))
    consistency=ef_consistency_loss(values['ef'],values['edv'],values['esv'])
    result['consistency']=str(float(consistency.detach()))
    result['focal']=str(float(focal_soft_loss(values['class_logits'],soft_class_targets(batch['ef'])).detach()))
    result['total']=str(float(full_joint_loss(values,batch,consistency_weight).detach()))
    return result


def replay_forward_failure(args):
    """One diagnostic epoch from latest saved none-arm boundary; no official run mutation."""
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from src.data.echonet_dataset import EchoNetDataset,AugmentConfig
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    root=Path(args.run_dir);last=root/'none/last.pt'
    if not last.exists():raise SystemExit('STOPPED: none/last.pt missing')
    state=torch.load(last,map_location='cpu',weights_only=True);cfg=state['protocol']['config'];dc=cfg['data'];tc=cfg['train']
    epoch=state['epoch']
    if file_hash(dc['file_list_csv'])!=state['protocol']['file_list_sha256']:raise SystemExit('FileList changed')
    private=root/f'forward_probe_after_epoch{epoch}'
    if private.exists():raise SystemExit('STOPPED: diagnostic directory already exists; preserve it')
    private.mkdir()
    class Centered(EchoNetDataset):
        def _sample_clip_indices(self,n):return centered_indices(n,self.num_frames,self.frame_stride)
    ds=Centered(cache_dir=dc['cache_dir'],file_list_csv=dc['file_list_csv'],split='TRAIN',num_frames=32,frame_stride=2,
                ef_thresholds=list(CUTOFFS),augment=AugmentConfig(temporal=False),seed=42)
    labels=class_targets(torch.tensor(ds.df['EF'].to_numpy(),dtype=torch.float32));counts=torch.bincount(labels,minlength=4)
    sampler=WeightedRandomSampler((1./counts.double())[labels],num_samples=len(ds),replacement=True,generator=torch.Generator().manual_seed(42+epoch))
    loader=DataLoader(ds,batch_size=4,sampler=sampler,num_workers=0,drop_last=False)
    m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).cuda();m.load_state_dict(state['model_state_dict'])
    opt=torch.optim.AdamW([{'params':list(m.head.parameters())+list(m.class_head.parameters()),'lr':tc['lr_head']},
                           {'params':m.backbone.parameters(),'lr':tc['lr_backbone']}],weight_decay=tc['weight_decay'])
    opt.load_state_dict(state['optimizer_state_dict'])
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=4096.);scaler.load_state_dict(state['scaler_state_dict'])
    deterministic_seed(42+epoch);ds.rng=np.random.default_rng(42000+epoch)
    for p in m.backbone.parameters():p.requires_grad_(epoch>=4)
    m.train()
    if epoch<4:m.backbone.eval()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mean=torch.tensor(R3D_MEAN,device='cuda').view(1,3,1,1,1);std=torch.tensor(R3D_STD,device='cuda').view(1,3,1,1,1)
    print(f'DIAGNOSTIC ONLY: replay at most epoch{epoch+1} from saved epoch{epoch}; original checkpoints/protocol untouched.',flush=True)
    print('Private failure data stays LOCAL. No VAL/TEST, no successful training output. Latest checkpoint SHA256',file_hash(last),flush=True)
    for step,batch in enumerate(loader):
        batch={k:batch[k].cuda() for k in ('clip','ef','edv','esv')};clip=(batch['clip']-mean)/std
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        buffers={k:v.detach().clone() for k,v in m.named_buffers()}
        try:
            safe_scaled_batch(m,opt,scaler,clip,batch,cfg['model']['ef_consistency_weight'],tc['grad_clip_norm'])
        except RuntimeError as error:
            # Restore exact preforward buffers before retaining and probing failure state.
            with torch.no_grad():
                for k,v in m.named_buffers():v.copy_(buffers[k])
            atomic_checkpoint({'model_state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict(),
                               'scaler_state_dict':scaler.state_dict(),'batch':batch,'cpu_rng':cpu_rng,'gpu_rng':gpu_rng,
                               'epoch':epoch+1,'step':step,'last_checkpoint_sha256':file_hash(last)},private/'failure_state.pt')
            report={'epoch':epoch+1,'step_zero_based':step,'error':str(error),
                    'input_finite':bool(torch.isfinite(clip).all()),
                    'target_finite':{k:bool(torch.isfinite(batch[k]).all()) for k in ('ef','edv','esv')}}
            print('FAILURE LOCATION',report,flush=True)
            for mode,amp in (('original_AMP',True),('full_FP32',False)):
                with torch.no_grad():
                    for k,v in m.named_buffers():v.copy_(buffers[k])
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',enabled=amp):out=m(clip)
                # Original loss was OUTSIDE autocast in diagnostic too; full_joint_loss uses
                # half output dtypes for consistency, while smooth-L1 promotes via fp32 labels.
                with torch.autocast(device_type='cuda',enabled=amp):
                    original=loss_breakdown(out,batch,cfg['model']['ef_consistency_weight'])
                with torch.autocast(device_type='cuda',enabled=False):
                    float_loss=loss_breakdown(out,batch,cfg['model']['ef_consistency_weight'],True)
                report[mode]={'original_arithmetic':original,'float32_loss_arithmetic':float_loss}
                print(mode,report[mode],flush=True)
            atomic_json(report,private/'failure_report.json')
            print('Stopped at captured failure; no optimizer update for offending batch. Send terminal output ONLY.',flush=True)
            return
        if step%200==0:print('Diagnostic finite steps:',step+1,flush=True)
    atomic_json({'status':'not reproduced in one diagnostic epoch','from_saved_epoch':epoch},private/'failure_report.json')
    print('No failure reproduced in one epoch. Stop; no further epochs or official checkpoint writes.',flush=True)


def verify_forward_fp32(args):
    """Exact saved failing state, FP32 forward/loss/gradients; no optimizer update/write."""
    from pathlib import Path
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    root=Path(args.run_dir);last=torch.load(root/'none/last.pt',map_location='cpu',weights_only=True)
    cfg=last['protocol']['config'];epoch=last['epoch']
    path=root/f'forward_probe_after_epoch{epoch}/failure_state.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['last_checkpoint_sha256']!=file_hash(root/'none/last.pt'):raise SystemExit('STOPPED: checkpoint changed since capture')
    m=R3DFourClass(pretrained=False,dropout_p=cfg['model']['dropout_p']).cuda();m.load_state_dict(state['model_state_dict'])
    m.train()
    for p in m.backbone.parameters():p.requires_grad_(True)
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    batch={k:v.cuda() for k,v in state['batch'].items()}
    mean=torch.tensor(R3D_MEAN,device='cuda').view(1,3,1,1,1);std=torch.tensor(R3D_STD,device='cuda').view(1,3,1,1,1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['gpu_rng'])
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['train']['lr_head'])
    loss,norm,out=safe_fp32_batch(m,opt,(batch['clip']-mean)/std,batch,
                                cfg['model']['ef_consistency_weight'],cfg['train']['grad_clip_norm'],do_step=False)
    diag=gradient_diagnostics(m)
    print('EXACT FAILURE FP32 GATE: epoch',state['epoch'],'step',state['step'],'loss',float(loss),'gradient norm before clipping',float(norm),'gradient diagnostics after clipping',diag,flush=True)
    print('Loss components',loss_breakdown(out,batch,cfg['model']['ef_consistency_weight'],True),flush=True)
    # Restore BatchNorm buffers touched by diagnostic forward; parameters have not stepped.
    m.load_state_dict(before)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    print('FP32 production-path gate PASS: finite forward/loss/gradients and clipping on captured state. No optimizer step or files written. This is not a guarantee for future batches.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--selftest',action='store_true');modes.add_argument('--pilot',action='store_true')
    modes.add_argument('--train-none-fp32',action='store_true')
    modes.add_argument('--train-arm-fp32',choices=[a for a in ARMS if a!='none'],metavar='ARM',
                       help='bounded FP32 single augmentation arm: geometric, mixup, cutmix or temporal')
    modes.add_argument('--train-five',action='store_true');modes.add_argument('--selftest-full',action='store_true')
    modes.add_argument('--diagnose-numerics',action='store_true')
    modes.add_argument('--capture-failure',action='store_true')
    modes.add_argument('--verify-saved-failure',action='store_true')
    modes.add_argument('--replay-forward-failure',action='store_true')
    modes.add_argument('--verify-forward-fp32',action='store_true')
    ap.add_argument('--config',default='configs/base.yaml');ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--warmup',type=int,default=5);ap.add_argument('--steps',type=int,default=10)
    ap.add_argument('--fp32',action='store_true',help='timing pilot only: disable autocast/scaling, not a full-training switch')
    ap.add_argument('--run-dir',default=None)
    ap.add_argument('--initial-from',default='results/fourclass_ablation_v1/initial.pt')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--acknowledge-protocol',action='store_true')
    a=ap.parse_args()
    if a.run_dir is None:a.run_dir='results/fourclass_fp32_v4' if a.train_arm_fp32 else ('results/fourclass_fp32_v3' if a.train_none_fp32 else 'results/fourclass_ablation_v2')
    if (a.train_none_fp32 or a.train_arm_fp32) and a.run_dir in ('results/fourclass_ablation_v1','results/fourclass_ablation_v2'):
        raise SystemExit('STOPPED: FP32 must use a separate namespace; preserve AMP history')
    if a.selftest:selftest()
    elif a.selftest_full:selftest_full()
    elif a.pilot:pilot(a)
    elif a.diagnose_numerics:diagnose_numerics(a)
    elif a.capture_failure:capture_training_failure(a)
    elif a.verify_saved_failure:verify_saved_failure(a)
    elif a.replay_forward_failure:replay_forward_failure(a)
    elif a.verify_forward_fp32:verify_forward_fp32(a)
    else:
        if a.fp32:raise SystemExit('Use --fp32 only with --pilot; bounded FP32 training uses --train-none-fp32 or --train-arm-fp32')
        full_experiment(a)


if __name__=='__main__':main()
