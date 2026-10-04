"""
R3D-18 with a heteroscedastic head: predicts EF and log(variance) jointly,
trained with Gaussian NLL. Unlike MC Dropout (which estimates epistemic
uncertainty -- what the model doesn't know), this targets aleatoric
uncertainty -- inherent ambiguity in the input itself (probe angle, image
quality, a genuinely hard-to-read heart). That's the likely dominant
source of error here, based on MC Dropout showing no error correlation.
"""
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class R3DHeteroscedastic(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_p: float = 0.3, backbone_dropout_p: float = 0.15):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r3d_18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.drop1 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop2 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop3 = nn.Dropout3d(p=backbone_dropout_p)
        self.drop4 = nn.Dropout3d(p=backbone_dropout_p)

        self.dropout = nn.Dropout(p=dropout_p)
        self.shared = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        # EF mean, EF log-variance, EDV, ESV -- 4 outputs instead of 3.
        # EDV/ESV stay deterministic (point estimates); only EF gets a
        # variance head, since EF is the clinical quantity we need
        # calibrated uncertainty for.
        self.out = nn.Linear(128, 4)

    def forward(self, clip: torch.Tensor) -> dict:
        x = self.backbone.stem(clip)
        x = self.backbone.layer1(x); x = self.drop1(x)
        x = self.backbone.layer2(x); x = self.drop2(x)
        x = self.backbone.layer3(x); x = self.drop3(x)
        x = self.backbone.layer4(x); x = self.drop4(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)

        feats = self.shared(feats)
        out = self.out(feats)

        ef_mean = out[:, 0]
        # Clamp log-variance to a sane range -- unconstrained NLL training
        # can otherwise drive variance to 0 or explode early in training.
        ef_logvar = torch.clamp(out[:, 1], min=-6.0, max=6.0)
        edv = out[:, 2]
        esv = out[:, 3]

        return {"ef": ef_mean, "ef_logvar": ef_logvar, "edv": edv, "esv": esv}


def gaussian_nll_loss(pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian with predicted mean and variance.
    Minimizing this jointly trains the mean to be accurate AND the variance
    to reflect how wrong the mean actually tends to be -- large predicted
    variance is only 'cheap' if the error there is genuinely large."""
    precision = torch.exp(-pred_logvar)
    return (0.5 * precision * (target - pred_mean) ** 2 + 0.5 * pred_logvar).mean()


def ef_consistency_loss(pred_ef: torch.Tensor, pred_edv: torch.Tensor, pred_esv: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    implied_ef = (pred_edv - pred_esv) / (pred_edv.clamp(min=eps)) * 100.0
    return torch.abs(pred_ef - implied_ef).mean()