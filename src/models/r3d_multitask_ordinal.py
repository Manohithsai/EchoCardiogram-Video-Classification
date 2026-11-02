"""R3D-18 multi-task model WITH the auxiliary ordinal head (pre-registered experiment).

NEW FILE. src/models/r3d_multitask.py is NOT edited: the baseline class is
imported and subclassed, and the forward stage-walk is copied so the ordinal
head taps the SAME pooled features the regression head consumes (the 512-d
vector after self.dropout). Everything else is identical to the baseline.

Checkpoint note: this class's state_dict is the baseline's plus ordinal_head.*
keys only. Loading an ordinal checkpoint into the plain baseline class works
with strict=False and drops exactly those keys (covered by
test_ordinal_integration.py). Whether the evaluation script tolerates that
depends on how it loads checkpoints; verify at integration time instead of
assuming.
"""
import torch

from src.models.r3d_multitask import R3DMultiTask  # baseline, unchanged
from src.models.ordinal_head import OrdinalHead


class R3DMultiTaskOrdinal(R3DMultiTask):
    """Baseline R3D-18 multi-task model plus the auxiliary ordinal head."""

    def __init__(self, pretrained: bool = True, dropout_p: float = 0.3, backbone_dropout_p: float = 0.15):
        super().__init__(pretrained=pretrained, dropout_p=dropout_p, backbone_dropout_p=backbone_dropout_p)
        # self.head[0] is Linear(512, 128); its in_features is the pooled size.
        # fork_rng preserves the CPU RNG state across the ordinal head's
        # initialization draws, so after construction the global RNG stream is
        # exactly where the plain baseline model leaves it. That keeps the
        # shuffle/dropout streams matched with the baseline arm at the same
        # seed (verified by test_ordinal_integration.py).
        with torch.random.fork_rng(devices=[]):
            self.ordinal_head = OrdinalHead(in_features=self.head[0].in_features)

    def forward(self, clip: torch.Tensor) -> dict:
        """Copied stage-walk from the baseline forward, plus the ordinal logits.
        clip: (B, 3, T, H, W) float32, pre-normalized by the caller."""
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
            "ordinal_logits": self.ordinal_head(feats),
        }
