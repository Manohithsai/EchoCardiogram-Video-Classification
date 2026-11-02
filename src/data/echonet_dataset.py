"""
Dataset class for EchoNet-Dynamic, reading from the cache built by
scripts/build_cache.py. Handles clip sampling, EF class labels, and
EDV/ESV auxiliary targets for the multi-task head.

Design notes:
- Labels come straight from FileList.csv (EF, ESV, EDV). No re-derivation,
  so predictions can be checked against the same ground truth EchoNet used.
- EF class thresholds are configurable (default reduced/mildly-reduced/normal),
  computed from the continuous EF at __getitem__ time -- never hardcode class
  counts up front, since they should be measured from the real data in EDA,
  not assumed.
- Augmentation is a set of named, independent flags (not a single "augment=True"
  switch) so the ablation study in the project plan can toggle them individually.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class AugmentConfig:
    geometric: bool = False   # small random shift/scale
    intensity: bool = False   # gamma/contrast/speckle-like noise
    temporal: bool = True     # random clip start + stride jitter (on by default; cheap and always valid for train)
    mixup: bool = False       # optional negative-result arm; see project notes on why this is questionable for EF


def ef_to_class(ef: float, thresholds: list) -> int:
    """thresholds=[40, 50] -> 0: EF<40 (reduced), 1: 40<=EF<50 (mildly reduced), 2: EF>=50 (normal)."""
    cls = 0
    for t in thresholds:
        if ef >= t:
            cls += 1
        else:
            break
    return cls


class EchoNetDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        file_list_csv: str,
        split: str,  # "TRAIN" | "VAL" | "TEST" (matches FileList.csv's Split column)
        num_frames: int = 32,
        frame_stride: int = 2,
        ef_thresholds: Optional[list] = None,
        augment: Optional[AugmentConfig] = None,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.ef_thresholds = ef_thresholds or [40, 50]
        self.augment = augment or AugmentConfig(temporal=(split == "TRAIN"))
        self.is_train = split == "TRAIN"
        self.rng = np.random.default_rng(seed)

        df = pd.read_csv(file_list_csv)
        df = df[df["Split"].str.upper() == split.upper()].reset_index(drop=True)

        missing = []
        for fn in df["FileName"]:
            if not (self.cache_dir / f"{Path(fn).stem}.npy").exists():
                missing.append(fn)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cached clips missing for split={split} "
                f"(e.g. {missing[:3]}). Run scripts/build_cache.py first."
            )

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _sample_clip_indices(self, total_frames: int) -> np.ndarray:
        span = self.num_frames * self.frame_stride
        stride = self.frame_stride

        if self.is_train and self.augment.temporal:
            stride = self.rng.choice([max(1, self.frame_stride - 1), self.frame_stride, self.frame_stride + 1])
            span = self.num_frames * stride

        if total_frames <= span:
            # Video shorter than requested span: loop-pad indices instead of
            # crashing. Rare, but happens with a few short clips in EchoNet.
            idx = (np.arange(self.num_frames) * stride) % total_frames
            return idx

        max_start = total_frames - span
        start = self.rng.integers(0, max_start + 1) if self.is_train else max_start // 2
        idx = start + np.arange(self.num_frames) * stride
        return idx

    def _apply_augment(self, clip: np.ndarray) -> np.ndarray:
        """clip: (T, H, W) uint8. Returns float32 clip in [0, 1], augmented if enabled."""
        clip = clip.astype(np.float32) / 255.0

        if self.is_train and self.augment.intensity:
            gamma = self.rng.uniform(0.85, 1.15)
            clip = np.clip(clip, 1e-4, 1.0) ** gamma
            noise = self.rng.normal(0, 0.02, size=clip.shape).astype(np.float32)
            clip = np.clip(clip + noise, 0.0, 1.0)

        if self.is_train and self.augment.geometric:
            shift_x = self.rng.integers(-4, 5)
            shift_y = self.rng.integers(-4, 5)
            clip = np.roll(clip, shift=(shift_y, shift_x), axis=(1, 2))

        return clip

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        stem = Path(row["FileName"]).stem
        arr = np.load(self.cache_dir / f"{stem}.npy")  # (T, H, W) uint8

        idx = self._sample_clip_indices(arr.shape[0])
        clip = arr[idx]  # (num_frames, H, W)
        clip = self._apply_augment(clip)  # (num_frames, H, W) float32 in [0,1]

        # (T, H, W) -> (C=3, T, H, W) for video models; channel is repeated,
        # not learned, since EchoNet frames are single-channel ultrasound.
        clip_t = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)  # (3, T, H, W)

        ef = float(row["EF"])
        edv = float(row["EDV"])
        esv = float(row["ESV"])
        ef_class = ef_to_class(ef, self.ef_thresholds)

        return {
            "clip": clip_t,
            "ef": torch.tensor(ef, dtype=torch.float32),
            "edv": torch.tensor(edv, dtype=torch.float32),
            "esv": torch.tensor(esv, dtype=torch.float32),
            "ef_class": torch.tensor(ef_class, dtype=torch.long),
            "filename": row["FileName"],
        }
