# Pre-registered spec: auxiliary ordinal head on R3D-18 (UPDATED 30 Oct 2026)

Status: REVIEWED by Claude 30 Oct 2026 - no design problems; the four required edits are applied below, marked [review 30 Oct]. NOT yet in effect: before the first run, this file must be saved on the laptop with a timestamp (its hash recorded outside the file). If the project is a git repo, a commit of this change alone is allowed; do not commit unrelated changes, and if it is not a git repo there is nothing to commit.

Provenance: the original spec was forwarded from Claude and finalized with five fixes. This update applies the three corrections agreed after the final review and records the decisions made since (Case B, seeds, run order, converter step, secondary readout KEEP, measured counts, GPU estimate). Claude's original wording is preserved where it stands; changes are marked [correction], [decision] or [review 30 Oct] (Claude's final-review edits).

Hypothesis (not a proven fix): adding a four-class ordinal task during training may improve the EF-number predictions, and so mild recall, without hurting Normal recall or MAE.

## 1. Baseline eligibility (decided from records, not scores) - DECIDED: Case B

The original rule: the original run counts as a matched baseline only if its seed is recorded and its script and config can be matched to the current ones. Otherwise use fresh baselines only. Its score plays no part in this decision.

[decision] Records check (inspection report, 30 Oct 2026): the original run's seed IS recorded (42, in the checkpoint's saved config and the W&B run's logged config), and the saved config matches the current base.yaml. But no training-time script version exists in any reachable record: the W&B run saved no code snapshot and no git metadata, the checkpoint stores no code fields, and dropout layers carry no weights, so the checkpoint cannot show which model code trained it. The config match alone cannot establish a script match (backbone dropout is a constructor default, invisible to the saved config). Under the spec's own rule, the original cannot count as a matched baseline seed: Case B.

The original run remains in the final report as a historical reference (VAL MAE 5.467, mild recall 27.1% at one centred clip), never as a comparator.

One optional record check remains: `git log --oneline -5 -- src/models/r3d_multitask.py src/train/train.py` on the laptop, if the project is a git repo. It could change the decision only by positively proving the training-time code (for example by showing the model file was unchanged between 3 Oct 2026 and now). The Case B decision rests on the records already reachable, not on any claim about what a git history would show.

## 2. Run budget

Case A (not in effect; kept for the record): original = baseline seed 1, plus 2 new baselines, 3 ordinal runs and 1 reproduction run, for 6 new runs. [correction] The reproduction run is only a diagnostic: it never decides eligibility, and it is EXCLUDED from the paired analysis. The seed-matched pairs are the original plus the 2 new baselines against the 3 ordinal runs, paired by seed value.

Case B (in effect): 3 fresh baselines plus 3 ordinal runs, 6 runs total. [correction] No run "reuses the original seed value" and there is no reproduction check in Case B: an unrecorded training-time script is precisely the Case B condition, so no fresh run can verify reproduction of the original, and the original seed value has no special status.

[decision] Seeds, recorded in advance before any run: 101, 202, 303. The same three seeds are used for both arms, paired by seed value (baseline 101 with ordinal 101, and so on).

[decision] Run order, recorded in advance: baseline 101, ordinal 101, baseline 202, ordinal 202, baseline 303, ordinal 303. Pairs run interleaved so the arms stay matched if a technical stop ever truncates the plan.

GPU preflight (spec's rule: read the per-epoch time from the original W&B run, compute runs x epochs x time) - DONE: the original run finished 30 epochs in 8963 s (about 299 s per epoch, a proxy that includes startup, validation passes and any pauses). Six runs at about 30 epochs each: about 14.9 hours. Two seeds per arm (4 runs): about 10.0 hours. Caveats: the ordinal head adds a small unmeasured per-step cost; actual epochs vary with early stopping (the original stopped at 30 of a configured 40).

[decision] 30 Oct 2026: the owner delegated the timing choice, and the adopted rule is: COMPLETE all three matched pairs, even if total runtime exceeds the ~14.9 h estimate. No fixed hard cap is set. If a technical problem forces a stop, the work pauses, the reason is recorded, and only complete seed pairs enter the paired analysis; any incomplete pair is disclosed. There is no two-pair fallback in the adopted plan: reducing to two pairs would be a new decision, allowed only before any results are seen. No decision is ever based on results.

## 3. Ordinal head and loss

Classes: <40, 40-50, 50-70 and >=70 (from ground-truth EF; the boundary convention is >= at each cut-off, matching the dataset's ef_to_class and the evaluator).

[correction] Class counts: TRAIN and VAL only. MEASURED 30 Oct 2026 with count_ef_groups.py: TRAIN n=7465 (<40 = 948, 40-50 = 724, 50-70 = 5542, >=70 = 251); VAL n=1288 (<40 = 156, 40-50 = 133, 50-70 = 963, >=70 = 36). No VAL group falls under 30; >=70 at 36 is the smallest. TEST group counts are NOT read for any design decision; they are reported only at the single final run. Any group with fewer than 30 validation videos is reported as exploratory.

Head: one shared logit g and ordered biases, with b1 = b0 - softplus(d1) and b2 = b1 - softplus(d2). The cumulative logits are z_k = g + b_k.

Loss: L_ord is the mean of BCE-with-logits over z_k against 1[y > k], computed with the numerically stable logits-based form. It is unweighted. Total: L = L_baseline + 0.3 x L_ord, with lambda fixed. Everything else is identical to the baseline.

Code tests (all laptop-verified 30 Oct 2026): bias ordering holds for random initialisations (softplus is positive by construction; strict ordering verified at typical randomized magnitudes, not claimed as an unconditional floating-point guarantee); with strongly correct logits (margins of +/-20) the loss is below 1e-6, not exactly zero; gradients stay finite at extreme logits; EF exactly 40/50/70 maps to classes 1/2/3; integration checks passed 7/7.

## 4. Evaluation protocol

All checkpoints: the same script, FP32, one centered 32-frame clip, validation split. Save predictions, checkpoint hash and prediction hash, and verify labels match across runs. No further TEST access during development. TEST was already used earlier in the project, so the final evaluation is a single frozen reporting step with that exposure disclosed, not an untouched independent test.

[decision] Converter step (new): the unchanged evaluator loads the plain baseline class with strict=True and would stop on the extra ordinal_head.* keys, so after each ordinal run, strip_ordinal_keys.py writes a NEW stripped checkpoint (the original is never modified; removed keys verified against the 5-key ordinal_head.* allowlist; retained keys verified zero-missing with equal shapes and dtypes against the current baseline class; config, epoch and val_mae preserved). The evaluator scores the stripped checkpoint only.

[decision] Equivalence check (two-level), run ONCE: after the first real ordinal run is converted, before ANY seed's stripped checkpoint is scored. That is the scope originally forwarded and it is kept unchanged; later conversions for the remaining seeds still pass through the converter's own per-file safety checks (5-key allowlist, zero-missing, shape and dtype verification, metadata preservation). The check itself:
(a) Weight level: exact tensor equality on all 124 baseline keys between the ordinal and stripped checkpoints. Any mismatch is a converter fault; stop.
(b) Prediction level: both checkpoints loaded, both models in eval() mode, on CPU (the pre-registered route; FP32 enforced with TF32 disabled on any device), over the FIRST 20 VAL filenames in FileList.csv order (fixed subset, no random selection). Pass requires max absolute EF difference <= 1e-4 AND zero flips of the 40/50 classifications. Exact identity is NOT demanded: GPU/cuDNN nondeterminism can move predictions slightly even for identical weights, so a small fixed tolerance is pre-registered. Any non-finite prediction stops the check. The stripped file's metadata (config, epoch, val_mae) must equal the ordinal checkpoint's, and normalisation is resolved on both files independently and must agree. A JSON record keeps both checkpoint hashes, the video list and the observed maximum difference. [review 30 Oct] This JSON record is shared with Claude for review before any checkpoint is scored; a failure pauses the work (section 5).

Primary comparison: mild recall from the EF regression output (thresholded at 40/50) of the model with the ordinal head, minus mild recall from the EF regression output of the baseline.

[decision] Secondary readout: KEEP, pre-registered now. score_ordinal_readout.py scores only the ordinal head's own four-class output, on VAL only under current authority (the frozen final TEST step is a separate explicitly approved step). Fixed outputs: per-group recall for the four groups (<40, 40-50, 50-70, >=70), overall four-class accuracy, the full confusion matrix and a per-video predictions file; groups under 30 flagged exploratory. The secondary readout is reported but is never promoted to the primary comparison. [review 30 Oct] Resolved at Claude's final review: the scorer prints the four-class outputs, and the merged Normal view is computed at REPORTING time by summing the 50-70 and >=70 rows and columns of the printed confusion matrix. This reporting rule is fixed now; no script change.

Also reported for every run: Reduced and Normal recall, MAE, RMSE, true R2, and EF<50 true and false flags.

## 5. Uncertainty and reporting

Repeating the same videos across seeds doesn't create independent videos, so:

Per seed pair: report the mild-recall difference with its own paired bootstrap interval, resampling videos with the same indices for both models.
Combined estimate: average each video's correct/incorrect result across the seeds within each arm, take the per-video difference, then do a cluster bootstrap that resamples videos (keeping all seeds for a video together).
What that interval covers: uncertainty from the video sample, for these trained models. It doesn't fully capture training randomness, which is why the spread across seeds is also shown, and with three seeds it is only indicative.
Same treatment for guardrails: Normal recall and MAE get the same per-pair and combined reporting.

[review 30 Oct] Bootstrap details, fixed now:
- Number of resamples: 10,000.
- Interval: 95% percentile interval.
- Bootstrap RNG seed: one fixed value, recorded now: 404.
- Resampling population per metric: mild recall resamples the 133 true-mild VAL videos, Normal recall resamples the 963 + 36 true-Normal VAL videos (the >=50 group, 999 videos), and MAE resamples all 1,288 VAL videos.
- Effect size: the report states the size of each difference, not only its direction and interval, because "consistently positive" does not require a minimum effect.

Descriptive label, fixed now:
"Consistently positive": the primary difference is positive in every seed pair, the combined interval excludes zero, and both guardrails hold.
"Inconclusive": mixed seed pairs or an interval that includes zero.
"Negative": a consistent drop or a failed guardrail.

Guardrails (means across seeds): Normal recall at least 92% and MAE no more than about 0.15 worse than the matched baseline.

[correction] A failed diagnostic (the equivalence check above, or any sanity check) pauses the work for investigation. It never silently changes the arms, the pairs or the analysis: switching anything based on a diagnostic outcome would reintroduce outcome-dependent selection.

## 6. Stopping rule

If the result isn't "consistently positive", record it, don't tune lambda, and move to EfficientNet-B3 + LSTM or the augmentation study.

[review 30 Oct] The final TEST evaluation (which checkpoint or ensemble, which threshold) is out of scope here and will be pre-registered separately, using VAL only, before any TEST access.

## 7. Known caveats carried from the inspection report

- Each of the 4 DataLoader workers inherits a copy of the dataset's seeded NumPy generator, so duplication of augmentation draws across workers within an epoch is possible; torch's worker seeding does not affect this NumPy generator. The mechanism is identical in both arms; its effect is disclosed, not assumed away.
- Full run-to-run determinism is NOT guaranteed: torch.use_deterministic_algorithms is never called and CUDA ops remain nondeterministic. Equality of the post-construction global CPU RNG state between the arms was tested at the same seed; identical full training-time random streams are not guaranteed beyond that tested point.
- [review 30 Oct] VAL is used twice: checkpoints are selected by VAL MAE, and the primary comparison is also computed on VAL. Both arms are treated identically, so the double use affects both arms in the same way; even so, the absolute VAL numbers are somewhat optimistic and are not independent of model selection.
- Both trainer copies force W&B offline (mode="offline" at wandb.init); nothing is uploaded. Syncing logs later is a separate explicit owner decision.
- Checkpoint writes happen only in per-seed run directories with a fail-if-exists guard; r3d18_best.pt is never touched. Each run directory gets a provenance.json at start (file hashes, full config, normalization constants, library versions, seed, wandb mode, timestamp).

## 8. Before running - checklist with status

1. Code and config summary settling eligibility: DONE 30 Oct 2026 (inspection report; Case B).
2. Run-count and GPU-time estimate: DONE 30 Oct 2026 (~14.9 h proxy for 6 runs). Timing rule: DONE 30 Oct 2026 - complete all three pairs, no fixed cap, technical stops pause with reasons recorded, analysis on complete pairs only (section 2).
3. Group counts, ordinal-head code and unit tests: DONE 30 Oct 2026 (counts measured; test_ordinal_head.py 6/6 checks, test_ordinal_integration.py 7/7 checks, test_strip_ordinal_keys.py 6/6 checks, test_ordinal_scoring.py 8/8 checks - 3 scorer + 1 preprocessing + 4 equivalence - all passed on the laptop).
4. Get-FileHash record of the 12 new files and 4 existing files (train.py, r3d_multitask.py, echonet_dataset.py, configs/base.yaml) on the laptop: DONE 30 Oct 2026 - all 16 hashes recorded verbatim in Appendix A. (The evaluator's hash can be recorded separately later if it becomes material.)
5. Library-version comparison: DONE 30 Oct 2026. The laptop's pip freeze shows torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3 and wandb 0.30.0 - all five key packages match the original run's recorded requirements. Scope: this compares those five key packages only; it is not a claim that the entire environment is identical to the training-time one.
6. This updated pre-registration saved on the laptop with a timestamp: OPEN. After saving, run Get-FileHash on the saved file and record the hash outside the file, so the timestamp is checkable later. Conditional commit per the review: only if the project is a git repo, and only this change - no unrelated changes.
7. Claude's final review: DONE 30 Oct 2026 - no design problems; the four wording edits are applied (marked [review 30 Oct]); the secondary readout, Case B with seeds 101/202/303, the interleaved run order, the complete-all-three-pairs no-cap rule, and the two-level once-only equivalence check are accepted as written.
8. Operational readiness at launch: OPEN. Laptop plugged in and sleep turned off (a run is about 2.5 hours; a sleep or thermal stop mid-run counts as a technical stop under section 2). Launch order per the review: baseline seed 101 first, then ordinal seed 101 - the run order recorded in section 2.

(Claude's review counted eight checklist items with seven done; this checklist folds the GPU estimate and the timing rule into one item (item 2), so its numbering differs. Every condition from the review is covered above.)


## Appendix A - Laptop file hashes (recorded verbatim 30 Oct 2026)

SHA-256 as returned by Get-FileHash on the laptop (D:\Echocardiogram Classification\echo-ef-project). All 16 entries well-formed (64 hex characters). The 12 new files were verified against the delivered artifacts: each laptop hash matches the delivered content after accounting for Windows line endings and, on four files, one trailing blank line added by the editor on save - no content differences. The 4 existing files are recorded as-is.

| File (relative to project root) | SHA-256 |
| --- | --- |
| src\models\ordinal_head.py | BDE7F203D2866D096E86C1F1E959CD685C7D459A0C44FE8A2D8E0E272B82351F |
| src\models\r3d_multitask_ordinal.py | 437CA8FD7ABFD39774BFCB7CC6D74FDB08F05257B988DBAD5BEB211AC8771C60 |
| src\train\train_ordinal.py | 391918DAB7C9386BB2F9F4346034D3A1C36444BFDFA691B47D5EAE60CB39E5DB |
| src\train\train_matched_baseline.py | 64F706786473F94A01BD46A0A4F17E5FAD681D81FB1C931CB01366E1FAD7EF40 |
| test_ordinal_head.py | E0ED901A4B9F7B382BF2B677F2E7FB85AA72203D6B2073BAE4181FDB17BAE8C0 |
| test_ordinal_integration.py | 830CAA3731F6033D9EE239FC3BAC2106651702932583AC875E4A988D4C6D8B04 |
| count_ef_groups.py | 202CE44DDE09C38CE662058BBE09F084A9374C82B9AF96644C024F85E3D4E95E |
| strip_ordinal_keys.py | BD1120E026418CFB8E0E4FA8CBE6970CCDF6F2E184FD817BAF489DCFD7BC31D7 |
| test_strip_ordinal_keys.py | 0E75B49531C941101E72D4C80620BC85658DADE0402372A892197792F2838C2F |
| score_ordinal_readout.py | 49CC87B4F74B12537DD39B6703B6540DB79AD746CE8DDB37E02425B0F09E3E5C |
| check_stripped_equivalence.py | 35EE4936E12952D25ACAC40B0179EAFEB7D27DC80E5A3C06FDB8A66FBBC6F4A9 |
| test_ordinal_scoring.py | 28C8FF6EA70E8ADF9D1D798533CFD9B2E0A6EFE0F63C2EC8346A142AB28B2B24 |
| src\train\train.py | F010615C7CC61FE0DDA72309AD4A3521FF28E82B29DB76CB183C23ECFB9577B3 |
| src\models\r3d_multitask.py | 15617B0439B299C17906F87BD62BE82A65E3F128D839AF9767152AFCF5BACA50 |
| src\data\echonet_dataset.py | 12118E66711A08F96F96E8F1286C2344D7ADF08982960BBEB309C48D24D84FC8 |
| configs\base.yaml | A749966BAB2A605BFA5F920037ED597FA2C82C5B3A7A8CA81561AA62102E5FF3 |
