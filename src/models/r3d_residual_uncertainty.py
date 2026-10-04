"""
Decoupled heteroscedastic uncertainty: the mean predictor (EF/EDV/ESV) is
frozen entirely, reusing the already-trained r3d18_mcdropout_best.pt. Only
a small new logvar head is trained, on top of frozen features, to predict
how large the FROZEN model's error tends to be. This avoids the variance
collapse seen when mean and logvar share a jointly-trained output layer.
"""
import torch
import torch.nn as nn

from src.models.r3d_multitask import R3DMultiTask


class R3DResidualUncertainty(nn.Module):
    def __init__(self, frozen_mean_model: R3DMultiTask, feat_dim: int = 128):
        super().__init__()
        self.mean_model = frozen_mean_model
        for p in self.mean_model.parameters():
            p.requires_grad = False
        self.mean_model.eval()

        # Small, separately trained head. Takes the same 128-dim shared
        # features the mean model's head uses (recomputed here since
        # R3DMultiTask doesn't expose intermediate features directly).
        self.logvar_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def extract_shared_features(self, clip: torch.Tensor) -> torch.Tensor:
        """Replicates R3DMultiTask's forward up through its shared 128-dim
        layer, under no_grad since this path is frozen."""
        m = self.mean_model
        with torch.no_grad():
            x = m.backbone.stem(clip)
            x = m.backbone.layer1(x); x = m.drop1(x)
            x = m.backbone.layer2(x); x = m.drop2(x)
            x = m.backbone.layer3(x); x = m.drop3(x)
            x = m.backbone.layer4(x); x = m.drop4(x)
            x = m.backbone.avgpool(x)
            feats = torch.flatten(x, 1)
            feats = m.dropout(feats)  # still eval mode -> dropout is a no-op here, fine
            shared_feats = m.head[0](feats)  # Linear(in_features, 128)
            shared_feats = m.head[1](shared_feats)  # ReLU
            ef_mean = m.head[3](m.head[2](shared_feats))[:, 0]  # full head output, take EF
        return shared_feats.detach(), ef_mean.detach()

    def forward(self, clip: torch.Tensor) -> dict:
        shared_feats, ef_mean = self.extract_shared_features(clip)
        logvar = self.logvar_head(shared_feats).squeeze(-1)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return {"ef": ef_mean, "ef_logvar": logvar}


def gaussian_nll_loss(pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-pred_logvar)
    return (0.5 * precision * (target - pred_mean) ** 2 + 0.5 * pred_logvar).mean()