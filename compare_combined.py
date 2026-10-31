"""Combined analysis for the Echo-EF ordinal experiment (VAL only).

Implements docs/ordinal_experiment_prereg.md section 5, with the implementation
clarifications accepted at review on 2026-10-31 and recorded in
docs/prereg_addendum_combined.md.

The section-5 text this follows:
  "Combined estimate: average each video's correct/incorrect result across the
   seeds within each arm, take the per-video difference, then do a cluster
   bootstrap that resamples videos (keeping all seeds for a video together)."
  "Same treatment for guardrails: Normal recall and MAE get the same per-pair
   and combined reporting."
  Labels: "Consistently positive" = the primary difference is positive in every
  seed pair, the combined interval excludes zero, and both guardrails hold.
  "Inconclusive" = mixed seed pairs or an interval that includes zero.
  "Negative" = a consistent drop or a failed guardrail.
  Guardrails (means across seeds): Normal recall at least 92% and MAE no more
  than about 0.15 worse than the matched baseline.
  Section 5 also fixes the bootstrap details for per-pair and combined alike:
  10,000 resamples, 95% percentile intervals, bootstrap RNG seed 404, and the
  per-metric resampling populations (133 mild / 999 Normal / 1,288 for MAE).

Addendum clarifications implemented here (see docs/prereg_addendum_combined.md):
  the MAE guardrail is pinned to exactly 0.15; the Normal guardrail is the
  ordinal arm's mean recall across seeds as an absolute floor; a per-pair
  difference of exactly zero is neither positive nor negative; combined MAE
  averages per-video absolute errors across seeds before differencing (no
  ensembling); seed-303 fingerprint constants stay PENDING until its scoring;
  a failed guardrail is "negative" even when the primary differences are
  positive, with the label components printed next to the label.

Reads the six evaluator result folders (baseline + ordinal for each seed pair).
Stops without computing anything if:
  - any pair-303 constant is still the PENDING placeholder,
  - any run is not VAL / fp32 / one centred clip / autocast off / tf32 disabled,
  - any checkpoint or prediction hash prefix differs from the recorded constants,
  - the six runs disagree on the evaluation setup (arch, sampling, normalisation,
    config and CSV hashes),
  - filenames or true EF labels differ across runs, or counts differ from the
    registered 1288 / 133 mild / 999 normal.

The interval covers video-sample uncertainty for these trained models only;
the spread across seed pairs is also shown. With three seeds it is indicative.

Run from the project root:
    python compare_combined.py --baseline-101 results\eval_split\val_matched_baseline101_fp32_k1 --ordinal-101 results\eval_split\val_ordinal101_stripped_fp32_k1 --baseline-202 results\eval_split\val_matched_baseline202_fp32_k1 --ordinal-202 results\eval_split\val_ordinal202_stripped_fp32_k1 --baseline-303 results\eval_split\val_matched_baseline303_fp32_k1 --ordinal-303 results\eval_split\val_ordinal303_stripped_fp32_k1 --tag combined
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Section 5 fixes these for per-pair and combined alike (addendum section 1).
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 404
EXPECTED = {"videos": 1288, "mild": 133, "normal": 999}  # registered VAL populations

# Recorded checkpoint/prediction hash prefixes per pair and arm (first 16 hex
# chars, as each metrics.json records them). PENDING placeholders stop the run.
PAIR_CONSTANTS = {
    "101": {"ckpt16": {"baseline": "b145586d7c32d6e1", "ordinal": "84be1d7f10c81670"},
            "pred16": {"baseline": "e1fa74864db3832f", "ordinal": "0c7d9547958bfb8e"}},
    "202": {"ckpt16": {"baseline": "6f9687f3abf26800", "ordinal": "fe1ccd6606a28b2a"},
            "pred16": {"baseline": "b7f2568bc91ba62f", "ordinal": "f8d69aabba1b9dce"}},
    "303": {"ckpt16": {"baseline": "PENDING-303", "ordinal": "PENDING-303"},
            "pred16": {"baseline": "PENDING-303", "ordinal": "PENDING-303"}},
}
PAIR_ORDER = ["101", "202", "303"]  # fixed: seed pairs are reported in this order

# Addendum items 1 and 2: "about 0.15" pinned to exactly 0.15; the Normal
# guardrail is the ordinal arm's mean recall across seeds as an absolute floor.
NORMAL_RECALL_MIN = 0.92
MAE_WORSE_MAX = 0.15


def stop(msg):
    raise SystemExit(f"STOPPED: {msg}")


def sha256_file(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def load_run(dir_path, pair, arm):
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
    if meta.get("tf32_disabled") is not True:
        stop(f"{d}: metrics.json does not record tf32_disabled=true; required for the fp32 protocol")
    ck16 = meta.get("checkpoint_sha256_16")
    want_ck = PAIR_CONSTANTS[pair]["ckpt16"][arm]
    if ck16 != want_ck:
        stop(f"{d}: checkpoint hash prefix {ck16!r} does not match the recorded pair-{pair} {arm} constant "
             f"({want_ck}); wrong checkpoint/folder - do not continue")
    recorded = meta.get("predictions_sha256_16")
    actual = sha256_file(npz, 16)
    if recorded != actual:
        stop(f"{d}: predictions.npz does not match the hash recorded in its metrics.json "
             f"(recorded {recorded!r}, current {actual}); inspect the files, do not rerun")
    want_pr = PAIR_CONSTANTS[pair]["pred16"][arm]
    if recorded != want_pr:
        stop(f"{d}: prediction hash prefix {recorded!r} does not match the recorded pair-{pair} {arm} constant "
             f"({want_pr}); wrong result folder - do not continue")
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
            "predictions_sha256_16": actual, "predictions_sha256_full": sha256_file(npz),
            "checkpoint": meta.get("checkpoint"), "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    for pair in PAIR_ORDER:
        ap.add_argument(f"--baseline-{pair}", required=True, help=f"baseline result folder for seed {pair}")
        ap.add_argument(f"--ordinal-{pair}", required=True, help=f"stripped-ordinal result folder for seed {pair}")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    # refuse early, before touching any result folder (addendum item 5)
    pending = [f"{p}-{a}" for p in PAIR_ORDER for a in ("baseline", "ordinal")
               if PAIR_CONSTANTS[p]["ckpt16"][a] == "PENDING-303"
               or PAIR_CONSTANTS[p]["pred16"][a] == "PENDING-303"]
    if pending:
        stop(f"seed-303 fingerprint constants are still PENDING placeholders for {', '.join(pending)}; "
             "fill them after seed-303 scoring, record the diff and hash, and review before running")

    out_dir = Path("results_pair_analysis") / args.tag
    if out_dir.exists():
        stop(f"{out_dir} already exists; refusing to overwrite - inspect, do not continue")

    runs = {}
    for pair in PAIR_ORDER:
        runs[(pair, "baseline")] = load_run(getattr(args, f"baseline_{pair}"), pair, "baseline")
        runs[(pair, "ordinal")] = load_run(getattr(args, f"ordinal_{pair}"), pair, "ordinal")

    # all six runs must come from the same evaluation setup
    first = runs[("101", "baseline")]["meta"]
    for key, r in runs.items():
        for field in ("arch", "sampling", "normalisation_mean", "normalisation_std",
                      "eval_config_sha256_16", "file_list_csv_sha256_16", "effective_data_config"):
            fv, rv = first.get(field), r["meta"].get(field)
            if fv is None or rv is None:
                stop(f"metrics.json is missing '{field}' on one side; cannot confirm all runs used the same setup")
            if fv != rv:
                stop(f"runs disagree on '{field}' between 101-baseline and {key}; not the same evaluation setup")
    if first["arch"] != "r3d18":
        stop(f"unexpected arch {first['arch']!r}; this comparison is for the R3D-18 pairs")

    # alignment: identical filenames and labels in the same order across all six runs
    names0, y0 = runs[("101", "baseline")]["names"], runs[("101", "baseline")]["y_true"]
    for key, r in runs.items():
        if not np.array_equal(r["names"], names0) or not np.array_equal(r["y_true"], y0):
            stop(f"alignment failed for {key}: filenames or labels differ; the runs are not directly comparable")
    print(f"PASS (alignment): {len(y0)} videos, same filenames and labels in the same order across all six runs")

    y = y0
    mild = (y >= 40.0) & (y < 50.0)
    normal = y >= 50.0
    counts = {"videos": len(y), "mild": int(mild.sum()), "normal": int(normal.sum())}
    for key, want in EXPECTED.items():
        if counts[key] != want:
            stop(f"VAL {key} count is {counts[key]}, expected the registered {want}; "
                 "the data does not match the frozen protocol - inspect, do not continue")

    # per-pair detail, kept so the combined run can be cross-checked against the
    # saved pair-101 and pair-202 analyses before its output is relied on
    pair_results = {}
    for pair in PAIR_ORDER:
        b, o = runs[(pair, "baseline")]["y_pred"], runs[(pair, "ordinal")]["y_pred"]
        pair_results[pair] = {
            "mild_baseline": float((((b >= 40.0) & (b < 50.0))[mild]).mean()),
            "mild_ordinal": float((((o >= 40.0) & (o < 50.0))[mild]).mean()),
            "mild_diff": float((((o >= 40.0) & (o < 50.0))[mild]).mean() - (((b >= 40.0) & (b < 50.0))[mild]).mean()),
            "normal_baseline": float((b >= 50.0)[normal].mean()),
            "normal_ordinal": float((o >= 50.0)[normal].mean()),
            "normal_diff": float((o >= 50.0)[normal].mean() - (b >= 50.0)[normal].mean()),
            "mae_baseline": float(np.abs(b - y).mean()),
            "mae_ordinal": float(np.abs(o - y).mean()),
            "mae_diff": float(np.abs(o - y).mean() - np.abs(b - y).mean()),
        }

    # combined: per-video averages across seeds within each arm, then per-video difference
    def combined(metric):
        if metric == "mild_recall":
            mask, rule = mild, lambda p: (p >= 40.0) & (p < 50.0)
        elif metric == "normal_recall":
            mask, rule = normal, lambda p: p >= 50.0
        else:
            mask, rule = np.ones(len(y), dtype=bool), None
        idx = np.where(mask)[0]
        if metric == "mae":
            b_avg = np.mean([np.abs(runs[(p, "baseline")]["y_pred"] - y)[idx] for p in PAIR_ORDER], axis=0)
            o_avg = np.mean([np.abs(runs[(p, "ordinal")]["y_pred"] - y)[idx] for p in PAIR_ORDER], axis=0)
        else:
            b_avg = np.mean([rule(runs[(p, "baseline")]["y_pred"])[idx].astype(float) for p in PAIR_ORDER], axis=0)
            o_avg = np.mean([rule(runs[(p, "ordinal")]["y_pred"])[idx].astype(float) for p in PAIR_ORDER], axis=0)
        diff = o_avg - b_avg
        return idx, float(b_avg.mean()), float(o_avg.mean()), diff

    # cluster bootstrap over videos (all seeds of a video stay together via the
    # per-video averages); one generator, fixed metric order mild, Normal, MAE
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    combined_results = {}
    for metric in ("mild_recall", "normal_recall", "mae"):
        idx, b_mean, o_mean, diff = combined(metric)
        n = len(idx)
        boot = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            boot[i] = diff[rng.integers(0, n, n)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        combined_results[metric] = {"population": n, "baseline": b_mean, "ordinal": o_mean,
                                    "diff": float(diff.mean()), "ci95": [float(lo), float(hi)]}

    # guardrails (addendum items 1 and 2). The combined MAE difference equals the
    # mean of the per-pair MAE differences by linearity, so one number is used.
    mean_normal_ordinal = float(np.mean([pair_results[p]["normal_ordinal"] for p in PAIR_ORDER]))
    mae_diff = combined_results["mae"]["diff"]
    guardrails = {"normal_recall": {"mean_ordinal_across_seeds": mean_normal_ordinal,
                                    "required_min": NORMAL_RECALL_MIN,
                                    "margin": mean_normal_ordinal - NORMAL_RECALL_MIN,
                                    "holds": mean_normal_ordinal >= NORMAL_RECALL_MIN},
                  "mae": {"diff": mae_diff,
                          "allowed_max_worse": MAE_WORSE_MAX,
                          "margin": MAE_WORSE_MAX - mae_diff,
                          "holds": mae_diff <= MAE_WORSE_MAX}}

    # Label logic with the addendum's precedence (item 6): check consistently
    # positive, then negative, then inconclusive. A failed guardrail is negative
    # even when the primary differences are positive; a per-pair difference of
    # exactly zero is neither positive nor negative, so such a result is
    # inconclusive unless a guardrail fails. The components are printed next to
    # the label so the precedence never hides how close the result was.
    mild_pairs = [pair_results[p]["mild_diff"] for p in PAIR_ORDER]
    lo, hi = combined_results["mild_recall"]["ci95"]
    every_positive = all(d > 0 for d in mild_pairs)
    every_negative = all(d < 0 for d in mild_pairs)
    guards_hold = guardrails["normal_recall"]["holds"] and guardrails["mae"]["holds"]
    if every_positive and lo > 0 and guards_hold:
        label = "consistently positive"
    elif every_negative or not guards_hold:
        label = "negative"
    else:
        label = "inconclusive"

    print(f"Combined analysis ({args.tag}): n={len(y)} videos, 3 seed pairs, cluster bootstrap over videos")
    for metric in ("mild_recall", "normal_recall", "mae"):
        r = combined_results[metric]
        print(f"  {metric}: baseline {r['baseline']:.3f} | ordinal {r['ordinal']:.3f} | "
              f"diff {r['diff']:+.3f} (95% CI {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})")
    print("Per-pair detail (cross-check against the saved pair analyses):")
    for p in PAIR_ORDER:
        r = pair_results[p]
        print(f"  pair {p}: mild baseline {r['mild_baseline']:.3f} ordinal {r['mild_ordinal']:.3f} "
              f"(diff {r['mild_diff']:+.3f}); Normal diff {r['normal_diff']:+.3f}; MAE diff {r['mae_diff']:+.3f}")
    print(f"Guardrails: Normal recall mean {mean_normal_ordinal:.3f} (min {NORMAL_RECALL_MIN}, "
          f"margin {guardrails['normal_recall']['margin']:+.3f}); MAE diff {mae_diff:+.3f} "
          f"(max worse {MAE_WORSE_MAX}, margin {guardrails['mae']['margin']:+.3f}) -> "
          f"{'both hold' if guards_hold else 'AT LEAST ONE FAILS'}")
    print(f"Label components: per-pair mild differences "
          f"{[round(d, 3) for d in mild_pairs]}; combined mild 95% CI [{lo:+.3f}, {hi:+.3f}]; "
          f"every pair positive: {every_positive}; every pair negative: {every_negative}; "
          f"guardrails hold: {guards_hold}")
    print(f"Label: {label}")
    print("Note: the interval covers video-sample uncertainty for these trained models; "
          "with three seeds it is indicative, not a full account of training randomness.")

    out_dir.mkdir(parents=True, exist_ok=False)
    meta_fields = ("tag", "split", "arch", "checkpoint", "checkpoint_sha256_16", "precision",
                   "autocast_active", "tf32_disabled", "clips_per_video", "sampling",
                   "normalisation_mean", "normalisation_std", "eval_config_sha256_16",
                   "file_list_csv_sha256_16", "effective_data_config")
    inputs = {f"{p}-{a}": {**{f: runs[(p, a)]["meta"].get(f) for f in meta_fields},
                           "dir": runs[(p, a)]["dir"],
                           "predictions_sha256_16": runs[(p, a)]["predictions_sha256_16"],
                           "predictions_sha256_full": runs[(p, a)]["predictions_sha256_full"]}
              for p in PAIR_ORDER for a in ("baseline", "ordinal")}
    out = {"tag": args.tag, "script_sha256": sha256_file(Path(__file__).resolve()),
           "numpy_version": np.__version__,
           "bootstrap": {"resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED,
                         "method": "cluster over videos, percentile 95"},
           "pairs": pair_results, "combined": combined_results, "guardrails": guardrails,
           "label": label, "inputs": inputs}
    (out_dir / "combined_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {out_dir}/combined_analysis.json")


if __name__ == "__main__":
    main()
