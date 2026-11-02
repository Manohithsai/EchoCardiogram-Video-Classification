"""Auxiliary ordinal head for the R3D-18 EF baseline (pre-registered experiment).

NEW FILE. The baseline training script is not modified. When training is
approved, integration happens through copied files (src/models/r3d_multitask_ordinal.py,
src/train/train_ordinal.py); the originals stay untouched.

Spec (docs/ordinal_experiment_prereg.md):
  - classes from ground-truth EF: <40, 40-50, 50-70, >=70  (>= at each cut-off,
    the same convention as the dataset's ef_to_class and the evaluator)
  - one shared linear layer produces a scalar logit g per video
  - ordered biases b0 > b1 > b2 via b1 = b0 - softplus(d1), b2 = b1 - softplus(d2)
  - cumulative logits z_k = g + b_k, k = 0, 1, 2
  - L_ord = mean BCE-with-logits(z_k, 1[y > k]), unweighted, stable logits form
  - total loss = L_baseline + 0.3 * L_ord  (lambda fixed at 0.3, never tuned)

Note on the ordering guarantee: softplus(d) is mathematically > 0, so the
ordering holds exactly in real arithmetic. In float32, an extremely negative d
(about -90 or below) can round softplus(d) to exactly 0 and produce a tie
(b_k == b_{k+1}) instead of a strict inequality. Typical parameter magnitudes
are far from that range; the unit tests check strict ordering for randomized
parameters and finite behaviour at extreme logits.
"""
import torch
from torch import nn
import torch.nn.functional as F

ORDINAL_THRESHOLDS = (40.0, 50.0, 70.0)
LAMBDA_ORD = 0.3  # fixed by the pre-registered spec; do not tune


def ef_to_ordinal_class(ef: torch.Tensor) -> torch.Tensor:
    """Continuous EF -> class index 0..3, using >= at 40/50/70.

    Matches EchoNetDataset.ef_to_class: EF exactly 40 is class 1 (mild),
    exactly 50 is class 2, exactly 70 is class 3.
    """
    cls = torch.zeros_like(ef, dtype=torch.long)
    for t in ORDINAL_THRESHOLDS:
        cls = cls + (ef >= t).long()
    return cls


class OrdinalHead(nn.Module):
    """CORAL-style ordinal head on pooled features."""

    def __init__(self, in_features: int):
        super().__init__()
        self.shared = nn.Linear(in_features, 1)  # produces the shared logit g
        self.b0 = nn.Parameter(torch.zeros(1))
        self.d1 = nn.Parameter(torch.zeros(1))
        self.d2 = nn.Parameter(torch.zeros(1))

    def biases(self):
        """Return (b0, b1, b2). Strictly ordered in real arithmetic because
        softplus > 0; see the module docstring for the float32 tie caveat."""
        b1 = self.b0 - F.softplus(self.d1)
        b2 = b1 - F.softplus(self.d2)
        return self.b0, b1, b2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, in_features) pooled R3D-18 features.
        Returns z: (batch, 3) cumulative logits z_k = g + b_k."""
        g = self.shared(features)  # (batch, 1)
        b0, b1, b2 = self.biases()
        return torch.cat([g + b0, g + b1, g + b2], dim=1)


def ordinal_loss(z: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits of z_k against 1[y > k], k = 0,1,2. Unweighted.

    Uses the numerically stable logits-based form (no manual log of sigmoid).
    """
    y = ef_to_ordinal_class(ef)                        # (batch,)
    k = torch.arange(3, device=z.device).unsqueeze(0)  # (1, 3)
    targets = (y.unsqueeze(1) > k).to(z.dtype)         # (batch, 3)
    return F.binary_cross_entropy_with_logits(z, targets, reduction="mean")


def total_loss(l_baseline: torch.Tensor, z: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
    """L = L_baseline + 0.3 * L_ord, lambda fixed by the spec."""
    return l_baseline + LAMBDA_ORD * ordinal_loss(z, ef)


def predict_class(z: torch.Tensor) -> torch.Tensor:
    """Secondary readout: number of p_k above 0.5 -> class index 0..3.

    p_k = sigmoid(z_k) estimates P(class > k). This readout is reported but
    never promoted to primary; the primary result stays the EF regression
    output thresholded at 40/50.
    """
    return (torch.sigmoid(z) > 0.5).sum(dim=1)
