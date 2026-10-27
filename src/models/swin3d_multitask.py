"""
Video Swin Transformer (Swin3D-T, Kinetics-400 pretrained, from torchvision) with the
same multi-task output interface as R3DMultiTask: forward(clip) -> {"ef", "edv", "esv"}.

clip: (B, 3, T, H, W), already normalised (see swin_norm_stats()).
"""
import torch
import torch.nn as nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def swin_norm_stats():
    """Mean/std that the pretrained torchvision weights expect (read from the weights'
    own transform, with an ImageNet fallback)."""
    try:
        tfm = Swin3D_T_Weights.KINETICS400_V1.transforms()
        mean, std = list(tfm.mean), list(tfm.std)
        if len(mean) == 3 and len(std) == 3:
            return mean, std
    except Exception:
        pass
    return IMAGENET_MEAN, IMAGENET_STD


class SwinMultiTask(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_p: float = 0.3):
        super().__init__()
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = swin3d_t(weights=weights)
        feat_dim = self.backbone.head.in_features  # 768 for Swin3D-T
        self.backbone.head = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 3),  # EF, EDV, ESV
        )
        with torch.no_grad():
            # start EF near the dataset mean so early epochs are not wasted
            self.head[-1].bias.copy_(torch.tensor([55.0, 0.0, 0.0]))

    def forward(self, x):
        z = self.head(self.backbone(x))
        return {"ef": z[:, 0], "edv": z[:, 1], "esv": z[:, 2]}