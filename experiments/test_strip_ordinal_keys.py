"""Tests for strip_ordinal_keys.py. NEW FILE. Uses only synthetic checkpoint
files in a temporary folder: no dataset is read, no split is evaluated, no
training happens, and no existing project file is touched.

Run from the project root:
    python test_strip_ordinal_keys.py
"""
import tempfile
from pathlib import Path

import torch
import yaml

from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_multitask_ordinal import R3DMultiTaskOrdinal
from src.eval.evaluate_split import (
    find_checkpoint_settings,
    compare_checkpoint_settings,
    resolve_normalisation,
)
from strip_ordinal_keys import strip_ordinal_checkpoint, ORDINAL_KEY_ALLOWLIST


def _make_ordinal_ckpt(path: Path, mutate=None) -> Path:
    """Save a checkpoint in exactly the shape the new trainers write."""
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["seed"] = 101  # as overridden by --seed in a real run
    model = R3DMultiTaskOrdinal(pretrained=False)
    sd = dict(model.state_dict())
    if mutate:
        sd = mutate(sd)
    torch.save({"model_state_dict": sd, "epoch": 7, "val_mae": 5.5, "config": cfg}, path)
    return path


def _expect_stop(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"stop message lacks {needle!r}: {e}"
        return
    raise AssertionError(f"expected SystemExit containing {needle!r}")


def test_strip_and_evaluator_paths() -> None:
    with tempfile.TemporaryDirectory() as td:
        src = _make_ordinal_ckpt(Path(td) / "ord.pt")
        out = Path(td) / "ord_for_eval.pt"
        prov = strip_ordinal_checkpoint(src, out)

        assert prov["removed_keys"] == sorted(ORDINAL_KEY_ALLOWLIST), prov["removed_keys"]
        assert out.is_file()

        # the stripped file loads with weights_only=True and strict=True into the baseline class
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        base = R3DMultiTask(pretrained=False)
        base.load_state_dict(ckpt["model_state_dict"])  # strict=True default: zero missing/unexpected
        assert set(ckpt["model_state_dict"]) == set(base.state_dict())
        assert ckpt["epoch"] == 7 and ckpt["val_mae"] == 5.5 and ckpt["config"]["seed"] == 101
        print("PASS (strip): 5 allowlisted keys removed, all baseline keys kept, "
              "strict=True load into the unchanged baseline class succeeds")

        # the evaluator's own provenance/config/norm paths behave as for a baseline checkpoint
        found = find_checkpoint_settings(ckpt)
        assert set(found) == {"num_frames", "frame_stride", "ef_thresholds"}, found
        data_cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))["data"]
        rows, mismatches = compare_checkpoint_settings(found, data_cfg)
        assert not mismatches, mismatches
        assert all(r["status"] == "match" for r in rows)
        mean, std, source, gaps = resolve_normalisation("r3d18", ckpt)
        assert source.startswith("R3D constants"), source
        assert len(gaps) == 1 and "not stored" in gaps[0]
        print("PASS (evaluator paths): num_frames/frame_stride/ef_thresholds found in "
              "checkpoint['config']['data'] and match the config; normalisation resolves to the "
              "R3D constants with the same recorded gap a baseline checkpoint produces")


def test_refusals() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        foreign = _make_ordinal_ckpt(td / "foreign.pt",
                                     lambda sd: {**sd, "some_other_module.weight": torch.zeros(1)})
        _expect_stop(lambda: strip_ordinal_checkpoint(foreign, td / "o1.pt"), "outside the ordinal allowlist")
        print("PASS (refusal): keys outside the allowlist are rejected")

        def drop_key(sd):
            sd = dict(sd)
            del sd[sorted(k for k in sd if not k.startswith("ordinal_head"))[0]]
            return sd
        incomplete = _make_ordinal_ckpt(td / "incomplete.pt", drop_key)
        _expect_stop(lambda: strip_ordinal_checkpoint(incomplete, td / "o2.pt"), "baseline keys are missing")
        print("PASS (refusal): a missing baseline key is rejected")

        plain = _make_ordinal_ckpt(td / "plain.pt",
                                   lambda sd: {k: v for k, v in sd.items()
                                               if k not in ORDINAL_KEY_ALLOWLIST})
        _expect_stop(lambda: strip_ordinal_checkpoint(plain, td / "o3.pt"), "no ordinal_head.* keys")
        print("PASS (refusal): a checkpoint without ordinal keys is sent to direct evaluation")

        src = _make_ordinal_ckpt(td / "ord.pt")
        existing = td / "exists.pt"
        existing.write_bytes(b"x")
        _expect_stop(lambda: strip_ordinal_checkpoint(src, existing), "refusing to overwrite")
        assert existing.read_bytes() == b"x"
        _expect_stop(lambda: strip_ordinal_checkpoint(src, src), "never overwritten")
        print("PASS (refusal): existing output and same-file output are both rejected; nothing overwritten")


if __name__ == "__main__":
    test_strip_and_evaluator_paths()
    test_refusals()
    print("All strip-adapter tests passed. Synthetic files only: no dataset read, "
          "no split evaluated, no training run.")

