"""Pair-101 analysis for the Echo-EF ordinal experiment (VAL only, descriptive).

Implements the frozen plan in docs/ordinal_experiment_prereg.md section 5:
mild-recall difference between the two arms with a paired bootstrap interval
(same resample indices for both models), 10,000 resamples, seed 404, 95%
percentile interval. Normal recall and MAE get the same treatment. Effect
sizes are reported, not only intervals.

Reads the two evaluator result folders (metrics.json + predictions.npz).
Stops without computing anything if:
  - the two runs are not aligned (filenames or true EF labels differ),
  - the predictions file does not match the hash its metrics.json recorded,
  - the run was not VAL / fp32 / one centred clip,
  - the data fails basic checks (lengths, duplicates, non-finite values),
  - the VAL group counts differ from the registered 1288 / 133 mild / 999 normal.

One pair is descriptive. Final labels and guardrails are decided across all
three pairs, not here.

Run from the project root:
    python compare_pair.py --baseline-dir results\eval_split\val_matched_baseline101_fp32_k1 --ordinal-dir results\eval_split\val_ordinal101_stripped_fp32_k1 --tag pair_101
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 404
EXPECTED = {"videos": 1288, "mild": 133, "normal": 999}  # registered VAL populations
# launch-log checkpoint and prediction hashes for pair 101 (first 16 hex chars,
# as metrics.json records them)
EXPECTED_CKPT16 = {"baseline": "168e7d732c5dc83d", "ordinal": "5b5532282067a41b"}
EXPECTED_PRED16 = {"baseline": "69a912f31c7ff551", "ordinal": "f5df0a08d5d50cdf"}


def sha256_file(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def stop(msg):
    raise SystemExit(f"STOPPED: {msg}")


def load_run(dir_path, arm):
    d = Path(dir_path)
    mj, npz = d / "metrics.json", d / "predictions.npz"
    if not mj.is_file():
        stop(f"metrics.json not found in {d}")
    if not npz.is_file():
        stop(f"predictions.npz not found in {d}")
    meta = json.loads(mj.read_text(encoding="utf-8"))
    for field, want in (("split", "VAL"), ("precision", "fp32"), ("clips_per_video", 1)):
        got = meta.get(field)
        if field == "clips_per_video":
            if not isinstance(got, int) or got != 1:
                stop(f"{d}: metrics.json has clips_per_video={got!r}, expected exactly 1 (one centred clip)")
        elif got != want:
            stop(f"{d}: metrics.json has {field}={got!r}, expected {want!r}")
    if meta.get("autocast_active") is not False:
        stop(f"{d}: metrics.json does not record autocast_active=false; this analysis requires a recorded fp32 run")
    ck16 = meta.get("checkpoint_sha256_16")
    if ck16 != EXPECTED_CKPT16[arm]:
        stop(f"{d}: checkpoint hash prefix {ck16!r} does not match the launch-log record for the {arm} arm "
             f"({EXPECTED_CKPT16[arm]}); wrong checkpoint or wrong folder - do not continue")
    recorded = meta.get("predictions_sha256_16")
    actual = sha256_file(npz, 16)
    if recorded != actual:
        stop(f"{d}: predictions.npz does not match the hash recorded in its metrics.json "
             f"(recorded {recorded!r}, current {actual}); inspect the files, do not rerun")
    if recorded != EXPECTED_PRED16[arm]:
        stop(f"{d}: prediction hash prefix {recorded!r} is not the recorded pair-101 {arm} run "
             f"({EXPECTED_PRED16[arm]}); wrong result folder - do not continue")
    if meta.get("tf32_disabled") is not True:
        stop(f"{d}: metrics.json does not record tf32_disabled=true; required for the fp32 protocol")
    data = np.load(npz, allow_pickle=False)
    for k in ("y_true", "y_pred", "file_names"):
        if k not in data:
            stop(f"{d}: predictions.npz is missing array '{k}'")
    y_true, y_pred, names = data["y_true"], data["y_pred"], data["file_names"].astype(str)
    if y_true.ndim != 1 or y_pred.ndim != 1 or names.ndim != 1 or not (len(y_true) == len(y_pred) == len(names)):
        stop(f"{d}: arrays must be 1-D and equal length "
             f"(got y_true {y_true.shape}, y_pred {y_pred.shape}, file_names {names.shape})")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        stop(f"{d}: non-finite values in labels or predictions; inspect the run, do not average over them")
    if len(names) == 0 or any(not s for s in names) or len(set(names.tolist())) != len(names):
        stop(f"{d}: empty or duplicate video filenames; the runs are not trustworthy for pairing")
    return {"dir": str(d), "y_true": y_true.astype(float), "y_pred": y_pred.astype(float),
            "names": names, "checkpoint_sha256_16": meta.get("checkpoint_sha256_16"),
            "predictions_sha256_16": actual, "checkpoint": meta.get("checkpoint"), "meta": meta}


def paired_ci(diffs, rng):
    n = len(diffs)
    boots = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        idx = rng.integers(0, n, n)  # one index set resamples both arms together
        boots[i] = diffs[idx].mean()
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def compare(baseline_dir, ordinal_dir, tag):
    out_dir = Path("results_pair_analysis") / tag
    if out_dir.exists():
        stop(f"{out_dir} already exists. An analysis under this tag was already recorded; "
             "inspect it before doing anything else. Do not rerun under a new name to get a different number.")
    base = load_run(baseline_dir, "baseline")
    ordn = load_run(ordinal_dir, "ordinal")

    # both runs must come from the same evaluation setup
    for field in ("arch", "sampling", "normalisation_mean", "normalisation_std",
                  "eval_config_sha256_16", "file_list_csv_sha256_16", "effective_data_config"):
        bv, ov = base["meta"].get(field), ordn["meta"].get(field)
        if bv is None or ov is None:
            stop(f"metrics.json is missing '{field}' on one side; cannot confirm both runs used the same setup")
        if bv != ov:
            stop(f"the two runs disagree on '{field}' ({bv!r} vs {ov!r}); they are not the same evaluation setup")
    if base["meta"]["arch"] != "r3d18":
        stop(f"unexpected arch {base['meta']['arch']!r}; this comparison is for the R3D-18 pair")

    if not (base["names"] == ordn["names"]).all():
        stop("the two runs disagree on video filenames or their order; do not compute differences")
    if not np.array_equal(base["y_true"], ordn["y_true"]):
        stop("the two runs disagree on the true EF labels; do not compute differences")
    print(f"PASS (alignment): {len(base['names'])} videos, same filenames and labels in the same order")

    y = base["y_true"]
    mild = (y >= 40.0) & (y < 50.0)
    normal = y >= 50.0
    counts = {"videos": len(y), "mild": int(mild.sum()), "normal": int(normal.sum())}
    for key, want in EXPECTED.items():
        if counts[key] != want:
            stop(f"VAL {key} count is {counts[key]}, expected the registered {want}; "
                 "the data does not match the frozen protocol - inspect, do not continue")
    if counts["mild"] == 0 or counts["normal"] == 0:
        stop("an EF group is empty; recalls are undefined")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {}

    def add(name, pop_mask, base_correct, ord_correct, unit):
        diff = ord_correct.astype(float) - base_correct.astype(float)
        m = {"population": int(pop_mask.sum()), "unit": unit,
             "difference_ordinal_minus_baseline": float(diff.mean()),
             "ci95": paired_ci(diff, rng)}
        if unit == "recall":
            m["baseline_recall"] = float(base_correct.mean())
            m["ordinal_recall"] = float(ord_correct.mean())
        metrics[name] = m

    b, o = base["y_pred"], ordn["y_pred"]
    add("mild_recall", mild, ((b >= 40.0) & (b < 50.0))[mild], ((o >= 40.0) & (o < 50.0))[mild], "recall")
    add("normal_recall", normal, (b >= 50.0)[normal], (o >= 50.0)[normal], "recall")
    mae_b, mae_o = np.abs(b - y), np.abs(o - y)
    metrics["mae"] = {"population": len(y), "unit": "mae_points",
                      "baseline_mae": float(mae_b.mean()), "ordinal_mae": float(mae_o.mean()),
                      "difference_ordinal_minus_baseline": float((mae_o - mae_b).mean()),
                      "ci95": paired_ci(mae_o - mae_b, rng)}

    report = {"tag": tag, "scope": "VAL, one seed pair, descriptive only",
              "bootstrap": {"resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED,
                            "interval": "95% percentile", "paired_indices": True},
              "counts": counts, "metrics": metrics,
              "inputs": {"baseline": {k: base[k] for k in ("dir", "checkpoint", "checkpoint_sha256_16", "predictions_sha256_16")},
                         "ordinal": {k: ordn[k] for k in ("dir", "checkpoint", "checkpoint_sha256_16", "predictions_sha256_16")}},
              "note": "Labels (consistently positive / inconclusive / negative) and guardrails "
                      "(Normal recall >= 92%, MAE within about 0.15) apply to means across all three "
                      "seed pairs, not to this single pair."}

    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Pair analysis ({tag}): n={counts['videos']}, mild n={counts['mild']}, normal n={counts['normal']}")
    for name in ("mild_recall", "normal_recall"):
        m = metrics[name]
        print(f"  {name}: baseline {m['baseline_recall']:.3f} | ordinal {m['ordinal_recall']:.3f} | "
              f"diff {m['difference_ordinal_minus_baseline']:+.3f} (95% CI {m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f})")
    m = metrics["mae"]
    print(f"  MAE: baseline {m['baseline_mae']:.3f} | ordinal {m['ordinal_mae']:.3f} | "
          f"diff {m['difference_ordinal_minus_baseline']:+.3f} (95% CI {m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f})")
    print("Single pair is descriptive; no final label or guardrail verdict yet.")
    print(f"Saved {out_dir / 'analysis.json'}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True, help="evaluator result folder for the matched baseline")
    p.add_argument("--ordinal-dir", required=True, help="evaluator result folder for the stripped ordinal run")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    compare(args.baseline_dir, args.ordinal_dir, args.tag)
