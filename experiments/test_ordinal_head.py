"""Unit tests for src/models/ordinal_head.py, required by the pre-registered spec.

NEW FILE. Run from the project root:
    python test_ordinal_head.py
No training, no data files, no existing project files are touched.
"""
import torch
import torch.nn.functional as F

from src.models.ordinal_head import (
    OrdinalHead,
    ordinal_loss,
    total_loss,
    ef_to_ordinal_class,
    predict_class,
    LAMBDA_ORD,
)


def test_bias_ordering_random_init(n_trials: int = 100) -> None:
    """(a) b0 > b1 > b2 with the bias parameters themselves randomized.

    softplus is mathematically > 0, so ordering holds in real arithmetic;
    these trials use typical-magnitude parameters. Extremely negative d
    (about -90 or below in float32) can round softplus to 0 and tie, which
    the spec's tests do not require handling.
    """
    for t in range(n_trials):
        torch.manual_seed(t)
        head = OrdinalHead(in_features=512)
        with torch.no_grad():  # vary b0, d1, d2 themselves, not only the linear layer
            head.b0.normal_()
            head.d1.normal_()
            head.d2.normal_()
        b0, b1, b2 = head.biases()
        assert (b0 > b1).all() and (b1 > b2).all(), f"ordering violated at trial {t}"
    print(f"PASS (a): bias ordering b0 > b1 > b2 holds for {n_trials} randomized bias settings")


def test_loss_near_zero_strongly_correct() -> None:
    """(b) loss < 1e-6 for strongly correct logits (margins of +/-20)."""
    ef = torch.tensor([20.0, 45.0, 60.0, 75.0])  # one video per class 0,1,2,3
    y = ef_to_ordinal_class(ef)
    k = torch.arange(3).unsqueeze(0)
    targets = (y.unsqueeze(1) > k).float()
    z = torch.where(targets > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    loss = ordinal_loss(z, ef)
    assert loss.item() < 1e-6, f"loss {loss.item():.3e} not below 1e-6"
    print(f"PASS (b): loss {loss.item():.3e} < 1e-6 for strongly correct logits (margins +/-20)")


def test_gradients_finite_extreme_logits() -> None:
    """(c) loss and gradients stay finite at extreme logits, both correctly
    signed (saturated, near-zero gradient) and incorrectly signed (large
    but finite gradient that still drives learning)."""
    z = torch.tensor([
        [1e4, 1e4, 1e4],     # correct, class 3
        [-1e4, -1e4, -1e4],  # correct, class 0
        [-1e4, -1e4, -1e4],  # WRONG for class 3 (targets all 1)
        [1e4, 1e4, 1e4],     # WRONG for class 0 (targets all 0)
    ], requires_grad=True)
    ef = torch.tensor([75.0, 20.0, 75.0, 20.0])
    loss = ordinal_loss(z, ef)
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    assert torch.isfinite(z.grad).all(), "logit gradients are not finite"
    wrong_grad = z.grad[2:]
    # dL/dz = (sigmoid(z) - y) / (batch * 3 thresholds) under mean reduction, so a
    # saturated incorrectly-signed logit keeps |grad| = 1/12 (exactly, in float32),
    # nonzero and still driving learning; a correctly-signed saturated logit is ~0.
    expected = 1.0 / (z.shape[0] * 3)
    assert (wrong_grad.abs() > expected / 2).all(), ("incorrectly-signed saturated logits should "
                                                     "keep a nonzero gradient")
    print(f"PASS (c): finite at extreme logits, correctly and incorrectly signed "
          f"(loss={loss.item():.3e}; wrong-signed |grad| = {wrong_grad.abs().mean().item():.4f}, "
          f"expected {expected:.4f} under mean reduction)")


def test_head_parameter_gradients_finite() -> None:
    """Gradients reaching the HEAD PARAMETERS stay finite, including with
    large-magnitude features that saturate the cumulative logits."""
    torch.manual_seed(0)
    head = OrdinalHead(in_features=4)
    for scale in (1.0, 1e3):
        head.zero_grad()
        feats = torch.tensor([[1.0, -1.0, 0.5, -0.5], [0.0, 0.25, -0.25, 1.0]]) * scale
        ef = torch.tensor([75.0, 20.0])
        loss = ordinal_loss(head(feats), ef)
        loss.backward()
        assert torch.isfinite(loss), f"loss not finite at feature scale {scale}"
        for name, p in head.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), \
                f"non-finite grad in {name} at feature scale {scale}"
    print("PASS (params): finite loss and finite head-parameter gradients at feature scales 1 and 1e3")


def check_class_convention() -> None:
    """Consistency check with the dataset/evaluator >= convention (not one of
    the three spec tests, but guards the readout against boundary drift)."""
    ef = torch.tensor([39.999, 40.0, 49.999, 50.0, 69.999, 70.0])
    expect = torch.tensor([0, 1, 1, 2, 2, 3])
    got = ef_to_ordinal_class(ef)
    assert torch.equal(got, expect), f"class convention mismatch: {got.tolist()} != {expect.tolist()}"
    print("PASS (convention): EF exactly 40/50/70 maps to classes 1/2/3, matching the evaluator")


def check_total_loss_lambda() -> None:
    """Sanity: total loss adds exactly 0.3 * L_ord to the baseline loss."""
    torch.manual_seed(0)
    z = torch.randn(5, 3)
    ef = torch.tensor([35.0, 42.0, 55.0, 72.0, 48.0])
    l_base = torch.tensor(1.234)
    combined = total_loss(l_base, z, ef)
    expect = l_base + LAMBDA_ORD * ordinal_loss(z, ef)
    assert torch.allclose(combined, expect), "total loss formula mismatch"
    print(f"PASS (formula): total = L_baseline + {LAMBDA_ORD} * L_ord")


if __name__ == "__main__":
    test_bias_ordering_random_init()
    test_loss_near_zero_strongly_correct()
    test_gradients_finite_extreme_logits()
    test_head_parameter_gradients_finite()
    check_class_convention()
    check_total_loss_lambda()
    print("All ordinal-head unit tests passed. No training was run.")

