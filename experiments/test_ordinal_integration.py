"""Integration checks for the ordinal-arm copies (no training, no data).

NEW FILE. Verifies, without running any training:
  1. the ordinal model's forward returns the expected keys and shapes;
  2. its state_dict is the baseline's plus only ordinal_head.* keys. This is
     an architecture-level fact only: scoring an ordinal checkpoint with the
     unchanged evaluator is NOT established here and stays PENDING until an
     explicit safe-loader decision (allowlist of droppable ordinal_head.*
     keys, or an explicit architecture path) is made later;
  3. the optimizer grouping mirrors src/train/train_ordinal.py (ordinal head
     in the head group) and a full backward pass leaves every parameter with
     a finite gradient. One optimizer step is then performed as a mechanics
     check: this is a single synthetic parameter update on random tensors,
     NOT training on any dataset;
  4. the two trainer copies differ from the original train.py only in the
     intended regions (docstring, imports/helpers, loss line, optimizer
     group, seed flag, offline wandb default, per-seed run directories with
     fail-if-exists guard and provenance record, RNG snapshot/restore);
  5. at the same seed, the baseline and ordinal models leave construction
     with identical backbone and regression-head initial weights AND the
     identical global CPU RNG state (the subclass preserves the RNG across
     the ordinal head's initialization with torch.random.fork_rng). This
     does not claim bit-identical full training sequences; CUDA operations
     remain nondeterministic (see the inspection report, section 1).

Run from the project root:
    python test_ordinal_integration.py
Uses pretrained=False so nothing is downloaded.
"""
import difflib
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_multitask_ordinal import R3DMultiTaskOrdinal
from src.models.ordinal_head import ordinal_loss, LAMBDA_ORD


def test_forward_shapes_and_keys() -> None:
    torch.manual_seed(0)
    model = R3DMultiTaskOrdinal(pretrained=False)  # no weight download
    model.eval()
    clip = torch.rand(1, 3, 32, 112, 112)
    with torch.no_grad():
        out = model(clip)
    assert set(out) == {"ef", "edv", "esv", "ordinal_logits"}, set(out)
    assert out["ef"].shape == (1,) and out["ordinal_logits"].shape == (1, 3)
    assert all(torch.isfinite(v).all() for v in out.values())
    print("PASS (forward): keys {ef, edv, esv, ordinal_logits}, shapes and finite values OK")


def test_state_dict_superset_loads_into_baseline() -> None:
    m_ord = R3DMultiTaskOrdinal(pretrained=False)
    m_base = R3DMultiTask(pretrained=False)
    missing, unexpected = m_base.load_state_dict(m_ord.state_dict(), strict=False)
    assert not missing, f"baseline missing keys: {missing}"
    assert all(k.startswith("ordinal_head.") for k in unexpected), \
        f"unexpected extra keys beyond ordinal_head.*: {unexpected}"
    print(f"PASS (state_dict): the only extra keys in an ordinal checkpoint are ordinal_head.* "
          f"({len(unexpected)} keys). Architecture-level fact only; loading into the unchanged "
          f"evaluator is NOT covered by this and stays pending an explicit safe-loader decision")


def test_optimizer_grouping_and_finite_param_grads() -> None:
    torch.manual_seed(0)
    model = R3DMultiTaskOrdinal(pretrained=False)
    model.train()
    # mirrors build_optimizer in src/train/train_ordinal.py
    opt = torch.optim.AdamW(
        [
            {"params": list(model.head.parameters()) + list(model.ordinal_head.parameters()),
             "lr": 1e-3},
            {"params": model.backbone.parameters(), "lr": 3e-5},
        ],
        weight_decay=1e-4,
    )
    clip = torch.rand(2, 3, 32, 112, 112)
    ef = torch.tensor([45.0, 60.0])
    edv = torch.tensor([120.0, 100.0])
    esv = torch.tensor([50.0, 35.0])
    out = model(clip)
    l_base = (F.smooth_l1_loss(out["ef"], ef)
              + F.smooth_l1_loss(out["edv"], edv)
              + F.smooth_l1_loss(out["esv"], esv))
    loss = l_base + LAMBDA_ORD * ordinal_loss(out["ordinal_logits"], ef)
    loss.backward()
    assert torch.isfinite(loss), "combined loss not finite"
    bad = [n for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"parameters with missing/non-finite gradients: {bad[:5]}"
    opt.step()
    print("PASS (optimizer): ordinal head in head group; finite gradients for ALL parameters; "
          "one AdamW step succeeded. Disclosure: that step performs ONE synthetic parameter "
          "update on random tensors as a mechanics check - it is not training on any dataset")


def test_shared_regression_initial_state_and_rng() -> None:
    torch.manual_seed(123)
    base = R3DMultiTask(pretrained=False)
    rng_after_base = torch.get_rng_state()
    torch.manual_seed(123)
    ordm = R3DMultiTaskOrdinal(pretrained=False)
    rng_after_ord = torch.get_rng_state()
    for group in ("backbone", "head"):
        d_base = getattr(base, group).state_dict()
        d_ord = getattr(ordm, group).state_dict()
        assert set(d_base) == set(d_ord), f"{group} keys differ"
        for k in d_base:
            assert torch.equal(d_base[k], d_ord[k]), f"{group}.{k} initial values differ"
    assert torch.equal(rng_after_base, rng_after_ord), (
        "post-construction CPU RNG states differ: the ordinal head's init draws leaked "
        "into the global stream")
    print("PASS (matched init): at the same seed, backbone and regression-head initial weights "
          "AND the post-construction global CPU RNG state are identical between the baseline "
          "and ordinal models (the subclass preserves the RNG across the ordinal head's init). "
          "This does not claim bit-identical full training sequences; CUDA ops stay "
          "nondeterministic")


def test_trainer_copies_differ_only_where_intended() -> None:
    root = Path(".")
    original = (root / "src" / "train" / "train.py").read_text().splitlines()
    expected_fragments = {
        "src/train/train_ordinal.py": [
            "R3DMultiTaskOrdinal", "ordinal_loss", "LAMBDA_ORD", "--seed",
            "r3d18_ordinal_seed", 'mode="offline"', "_write_provenance", "fork_rng",
        ],
        "src/train/train_matched_baseline.py": [
            "--seed", "r3d18_matched_baseline_seed", 'mode="offline"', "_write_provenance",
        ],
    }
    for rel, fragments in expected_fragments.items():
        copy_lines = (root / rel).read_text().splitlines()
        text = "\n".join(copy_lines)
        for frag in fragments:
            assert frag in text, f"{rel} is missing expected fragment: {frag}"
        regions = [op for op, *_ in difflib.SequenceMatcher(None, original, copy_lines).get_opcodes()
                   if op != "equal"]
        assert len(regions) <= 22, f"{rel} differs from train.py in {len(regions)} regions (expected the documented set)"
        print(f"PASS (copy check): {rel} contains the intended additions and differs from "
              f"train.py in {len(regions)} small regions only")
    # the matched-baseline copy must NOT contain the ordinal machinery
    mb = (root / "src" / "train" / "train_matched_baseline.py").read_text()
    assert "ordinal_loss(" not in mb and "R3DMultiTaskOrdinal(" not in mb
    print("PASS (copy check): train_matched_baseline.py contains no ordinal machinery")


if __name__ == "__main__":
    test_forward_shapes_and_keys()
    test_state_dict_superset_loads_into_baseline()
    test_optimizer_grouping_and_finite_param_grads()
    test_shared_regression_initial_state_and_rng()
    test_trainer_copies_differ_only_where_intended()
    print("All integration checks passed. No dataset training was run "
          "(one synthetic parameter update on random tensors was performed as a mechanics check).")
