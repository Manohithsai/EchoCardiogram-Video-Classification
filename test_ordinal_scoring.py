"""Tests for score_ordinal_readout.py and check_stripped_equivalence.py.

NEW FILE. Uses a tiny SYNTHETIC dataset (six fake cached clips and a fake
FileList.csv in a temporary folder): no real data is read, no split is
evaluated, no training happens, and no existing project file is touched.

Run from the project root:
    python test_ordinal_scoring.py
"""
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from src.eval.evaluate_split import R3D_MEAN, R3D_STD
from src.models.r3d_multitask import R3DMultiTask
from src.models.r3d_multitask_ordinal import R3DMultiTaskOrdinal
from strip_ordinal_keys import strip_ordinal_checkpoint
from score_ordinal_readout import score
from check_stripped_equivalence import check

FAKE_EFS = [35.0, 45.0, 48.0, 60.0, 65.0, 75.0]  # classes 0, 1, 1, 2, 2, 3
EXPECTED_GROUP_N = {"<40": 1, "40-50": 2, "50-70": 2, ">=70": 1}


def _make_fake_project(td: Path) -> Path:
    """Six loop-padded fake VAL clips + FileList.csv + config yaml."""
    cache = td / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, ef in enumerate(FAKE_EFS):
        name = f"FAKE_{i:03d}.avi"
        np.save(cache / f"FAKE_{i:03d}.npy", rng.integers(0, 256, size=(40, 112, 112), dtype=np.uint8))
        edv, esv = 120.0, 120.0 - ef * 1.2
        rows.append((name, ef, esv, edv, "VAL"))
    csv_path = td / "FileList.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "EF", "ESV", "EDV", "Split"])
        w.writerows(rows)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["seed"] = 101
    cfg["data"]["cache_dir"] = str(cache)
    cfg["data"]["file_list_csv"] = str(csv_path)
    cfg_path = td / "fake_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def _make_ordinal_ckpt(path: Path) -> Path:
    torch.manual_seed(101)
    model = R3DMultiTaskOrdinal(pretrained=False)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["seed"] = 101
    torch.save({"model_state_dict": model.state_dict(), "epoch": 7, "val_mae": 5.5, "config": cfg}, path)
    return path


def _expect_stop(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"stop message lacks {needle!r}: {e}"
        return
    raise AssertionError(f"expected SystemExit containing {needle!r}")


def test_scorer() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg_path = _make_fake_project(td)
        ckpt = _make_ordinal_ckpt(td / "ord.pt")
        cwd = os.getcwd()
        os.chdir(td)  # scorer writes results_ordinal_readout/<tag> under cwd
        try:
            m1 = score(str(cfg_path), str(ckpt), "t1")
        finally:
            os.chdir(cwd)
        assert m1["n_videos"] == 6
        confusion = m1["confusion_true_rows_pred_cols"]
        assert sum(sum(r) for r in confusion) == 6
        assert {k: v["n"] for k, v in m1["per_group"].items()} == EXPECTED_GROUP_N
        for i, name in enumerate(["<40", "40-50", "50-70", ">=70"]):
            n_i = sum(confusion[i])
            expect = (confusion[i][i] / n_i) if n_i else None
            assert m1["per_group"][name]["recall"] == expect
        assert abs(m1["overall_accuracy"] - sum(confusion[i][i] for i in range(4)) / 6) < 1e-12
        preds = list(csv.reader(open(td / "results_ordinal_readout" / "t1" / "predictions.csv")))
        assert len(preds) == 7 and preds[0] == ["filename", "ef", "true_class", "pred_class"]
        assert all(g["exploratory"] for g in m1["per_group"].values())  # all groups < 30 here
        pp = m1["preprocessing"]
        assert pp["normalization_mean"] == R3D_MEAN and pp["normalization_std"] == R3D_STD
        assert pp["precision"] == "fp32" and "centred" in pp["sampling"]
        print("PASS (scorer): metrics, confusion, per-group recalls, exploratory flags, "
              "preprocessing record and predictions.csv are all consistent on the synthetic set")

        # determinism in eval mode: a second run under a new tag gives identical predictions
        os.chdir(td)
        try:
            m2 = score(str(cfg_path), str(ckpt), "t2")
        finally:
            os.chdir(cwd)
        p1 = [r[3] for r in preds[1:]]
        p2 = [r[3] for r in csv.reader(open(td / "results_ordinal_readout" / "t2" / "predictions.csv"))][1:]
        assert p1 == p2, "same checkpoint and eval mode must give identical predictions"
        print("PASS (scorer): repeat run is deterministic in eval mode")

        # existing tag refused; stripped (headless) checkpoint refused
        strip_ordinal_checkpoint(ckpt, td / "stripped.pt")
        os.chdir(td)
        try:
            _expect_stop(lambda: score(str(cfg_path), str(ckpt), "t1"), "already exists")
            _expect_stop(lambda: score(str(cfg_path), str(td / "stripped.pt"), "t3"),
                         "does not match the ordinal architecture")
        finally:
            os.chdir(cwd)
        print("PASS (scorer): existing result folder and headless checkpoint are both refused")


def test_scorer_normalizes_inputs() -> None:
    """The scorer must feed NORMALIZED clips, not the dataset's raw [0,1] values."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg_path = _make_fake_project(td)
        ckpt = _make_ordinal_ckpt(td / "ord.pt")
        captured = {}
        orig_forward = R3DMultiTaskOrdinal.forward

        def spy(self, clip):
            if "first_input" not in captured:
                captured["first_input"] = clip[0].detach().cpu().clone()  # first video, first batch
            return orig_forward(self, clip)

        R3DMultiTaskOrdinal.forward = spy
        cwd = os.getcwd()
        os.chdir(td)
        try:
            score(str(cfg_path), str(ckpt), "spy")
        finally:
            os.chdir(cwd)
            R3DMultiTaskOrdinal.forward = orig_forward

        from src.data.echonet_dataset import EchoNetDataset, AugmentConfig
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        ds = EchoNetDataset(cache_dir=cfg["data"]["cache_dir"], file_list_csv=cfg["data"]["file_list_csv"],
                            split="VAL", num_frames=cfg["data"]["num_frames"],
                            frame_stride=cfg["data"]["frame_stride"],
                            ef_thresholds=cfg["data"]["ef_thresholds"],
                            augment=AugmentConfig(temporal=False), seed=cfg["seed"])
        raw = ds[0]["clip"]  # (3, T, H, W) in [0, 1]
        mean_t = torch.tensor(R3D_MEAN).view(3, 1, 1, 1)
        std_t = torch.tensor(R3D_STD).view(3, 1, 1, 1)
        expected = (raw - mean_t) / std_t
        got = captured["first_input"]
        assert torch.allclose(got, expected, atol=1e-6), "scorer input is not the normalized clip"
        assert not torch.allclose(got, raw, atol=1e-3), "scorer fed the UNNORMALIZED [0,1] clip"
        print("PASS (preprocessing): scorer input equals the per-channel normalized clip "
              "(and is not the raw [0,1] clip)")


def test_equivalence() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg_path = _make_fake_project(td)
        ord_ckpt = _make_ordinal_ckpt(td / "ord.pt")
        strip_ordinal_checkpoint(ord_ckpt, td / "stripped.pt")
        rec = check(str(ord_ckpt), str(td / "stripped.pt"), str(cfg_path),
                    str(td / "equiv.json"), device="cpu")
        assert rec["result"] == "PASS" and rec["threshold_flips_40_50"] == 0
        assert rec["n_videos"] == 6  # fixed-subset logic takes min(20, available)
        assert rec["max_abs_ef_diff"] <= 1e-4
        saved = json.load(open(td / "equiv.json"))
        assert saved["ordinal_sha256"] and saved["stripped_sha256"] and len(saved["videos"]) == 6
        assert saved["precision"] == "fp32" and saved["preprocessing"]["normalization_mean"] == R3D_MEAN
        print(f"PASS (equivalence): exact weight equality and prediction agreement "
              f"(max |EF diff| {rec['max_abs_ef_diff']:.1e}) on the synthetic set, "
              f"preprocessing/device/precision recorded")

        # tampered stripped checkpoint must fail the weight-level check
        ck = torch.load(td / "stripped.pt", map_location="cpu", weights_only=True)
        k = sorted(ck["model_state_dict"])[0]
        ck["model_state_dict"][k] = ck["model_state_dict"][k] + 1e-3
        torch.save(ck, td / "tampered.pt")
        _expect_stop(lambda: check(str(ord_ckpt), str(td / "tampered.pt"), str(cfg_path),
                                   str(td / "equiv2.json"), device="cpu"), "weight mismatch")
        print("PASS (equivalence): a tampered stripped checkpoint fails the exact weight check")

        # metadata tamper in the stripped file must stop the check
        ck = torch.load(td / "stripped.pt", map_location="cpu", weights_only=True)
        ck["epoch"] = (ck.get("epoch") or 0) + 1
        torch.save(ck, td / "tampered_meta.pt")
        _expect_stop(lambda: check(str(ord_ckpt), str(td / "tampered_meta.pt"), str(cfg_path),
                                   str(td / "equiv_meta.json"), device="cpu"), "metadata mismatch")
        print("PASS (equivalence): tampered metadata in the stripped file stops the check")

        # a non-finite prediction must stop the check (NaN must not slip past max comparisons)
        orig_forward = R3DMultiTask.forward

        def nan_forward(self, clip):
            return {"ef": torch.full((clip.shape[0],), float("nan")),
                    "edv": torch.zeros(clip.shape[0]), "esv": torch.zeros(clip.shape[0])}

        R3DMultiTask.forward = nan_forward
        try:
            _expect_stop(lambda: check(str(ord_ckpt), str(td / "stripped.pt"), str(cfg_path),
                                       str(td / "equiv3.json"), device="cpu"), "non-finite")
        finally:
            R3DMultiTask.forward = orig_forward
        print("PASS (equivalence): a non-finite prediction stops the check")


if __name__ == "__main__":
    test_scorer()
    test_scorer_normalizes_inputs()
    test_equivalence()
    print("All scoring/equivalence tests passed. Synthetic data only: no real dataset read, "
          "no split evaluated, no training run.")
