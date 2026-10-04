"""
Frame-level CNN baseline: a pretrained ResNet-18 applied per-frame,
then temporal mean-pooling. This tests whether full 3D video modeling
(R3D) actually beats a much simpler approach.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FrameCNNMultiTask(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_p: float = 0.3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 3),  # [EF, EDV, ESV]
        )

    def forward(self, clip: torch.Tensor) -> dict:
        """clip: (B, 3, T, H, W) float32. Each frame goes through ResNet
        independently, then features are mean-pooled over time."""
        B, C, T, H, W = clip.shape
        frames = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
        feats = self.backbone(frames)  # (B*T, in_features)
        feats = feats.reshape(B, T, -1).mean(dim=1)  # (B, in_features) -- temporal mean pool
        feats = self.dropout(feats)
        out = self.head(feats)
        return {"ef": out[:, 0], "edv": out[:, 1], "esv": out[:, 2]}

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()