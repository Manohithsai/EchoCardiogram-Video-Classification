"""
R3D-18 (Kinetics-400 pretrained) with a multi-task regression head.

Predicts EF, EDV, ESV jointly. Dropout is now placed both inside the
backbone (after each residual stage) and in the head, so MC Dropout at
inference time actually samples diverse feature paths -- head-only
dropout was found to produce an uncertainty signal uncorrelated with
real error (see results/mc_dropout/).
"""
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class R3DMultiTask(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_p: float = 0.3, backbone_dropout_p: float = 0.15):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r3d_18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Spatial-temporal dropout after each residual stage. Dropout3d zeroes
        # whole feature channels (not individual voxels), which is the right
        # granularity for conv feature maps -- plain Dropout would barely
        # perturb spatially correlated activations.
        self.drop1 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop2 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop3 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop4 = nn.Dropout3d(p=backbone_dropout_p)

        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 3),  # [EF, EDV, ESV]
        )

    def forward(self, clip: torch.Tensor) -> dict:
        """clip: (B, 3, T, H, W) float32, pre-normalized by the caller.
        Manually walks the backbone's stages (instead of calling
        self.backbone(clip) directly) so dropout can be inserted between them."""
        x = self.backbone.stem(clip)
        x = self.backbone.layer1(x)
        x = self.drop1(x)
        x = self.backbone.layer2(x)
        x = self.drop2(x)
        x = self.backbone.layer3(x)
        x = self.drop3(x)
        x = self.backbone.layer4(x)
        x = self.drop4(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)

        feats = self.dropout(feats)
        out = self.head(feats)
        return {
            "ef": out[:, 0],
            "edv": out[:, 1],
            "esv": out[:, 2],
        }

    def enable_mc_dropout(self):
        """Call before inference to keep every dropout layer (backbone and
        head) active, while BatchNorm and everything else stays in eval mode."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()


def ef_consistency_loss(pred_ef: torch.Tensor, pred_edv: torch.Tensor, pred_esv: torch.Tensor) -> torch.Tensor:
    """Penalizes predictions where EF disagrees with (EDV - ESV) / EDV."""
    eps = 1e-4
    implied_ef = (pred_edv - pred_esv) / (pred_edv.clamp(min=eps)) * 100.0
    return torch.abs(pred_ef - implied_ef).mean()