
## Preflight - 2026-10-30 02:38:21 +05:30
- Preflight READY; checkpoint item PARTIAL (prefix match, full hash recorded).
- Pre-registration SHA-256: 6A90D9ECCC648C64429B562BE08D4A5B59BEC7C6816FD0562C2940BA1B76481C
- Prereg-only commit: 5da82cfed4589fe0c8143a2329f69acd55c29b16
- Historical checkpoint SHA-256: 7EA9C83359E3918019A99057BC7154CE1D66437AC4BC28F55BB25F7307DC194A
- Earlier record held only prefix 7ea9c83359e39180, which matches. Full hash first recorded 30 Oct 2026; unchanged since 3 Oct cannot be fully verified. Historical reference only.
- Historical VAL MAE 5.467 and mild recall 27.1% come from saved evaluation outputs, not a fresh re-run.
- Checklist item 6 complete; operational readiness checked during preflight, to be reconfirmed at launch.
- Training has not started as of this entry.

- Baseline seed 101 launch attempted: 2026-10-30 02:39:59 +05:30. Checklist item 8 confirmed by owner.

## Baseline seed 101 complete - logged 2026-10-30 10:50:26 +05:30
- Offline run: offline-run-20261030_024006-j3eo61rf; provenance written 2026-10-30 02:40:13 +05:30.
- Actual finish time: unknown. Completion output reported in chat on 2026-10-30 at 10:47 +05:30.
- Completed 40 epochs. Best epoch: 37. Best validation MAE: 5.332 (rounded script output).
- Final epoch validation MAE: 5.53476; this is not the best checkpoint's score.
- Files confirmed: checkpoints/r3d18_matched_baseline_seed101/best.pt (133007041 bytes) and provenance.json (2042 bytes).
- Best checkpoint saved at 05:49:03 +05:30; this is not the training finish time.
- Four initial non-finite-loss warnings shown in the supplied log; no further such warnings shown after epoch 1.
- No standalone evaluation or TEST access performed during this run.

- Ordinal seed 101 launch attempted: 2026-10-30 22:38:23 +05:30. Owner confirmed plugged in and sleep off.

## Ordinal seed 101 complete - logged 2026-10-31 01:44:24 +05:30
- Offline run: offline-run-20261030_223830-9r436e0q; provenance written 2026-10-30 22:38:37 +05:30.
- Actual finish time: unknown. Completion output reported in chat on 2026-10-31 at 01:40 +05:30.
- Early stopping at epoch 32 of 40 (6 epochs without improvement). Best epoch: 26. Best validation MAE: 5.491 (rounded script output).
- Final epoch (32) validation MAE: 5.59391; this is not the best checkpoint's score.
- Files confirmed: checkpoints/r3d18_ordinal_seed101/best.pt (133010600 bytes) and provenance.json (2248 bytes).
- Best checkpoint saved at 2026-10-31 01:03:21 +05:30; this is not the training finish time.
- Four initial non-finite-loss warnings shown in the supplied log; no further such warnings shown after epoch 1.
- No standalone scoring or TEST access reported. Training included validation as shown above.

## Equivalence gate and baseline hash - logged 2026-10-31 03:05:17 +05:30
- Baseline seed 101: checkpoints/r3d18_matched_baseline_seed101/best.pt; SHA-256 B145586D7C32D6E1528FC79B520CE4F76685BE397CDC41E05527BDD81E6E9914; 133007041 bytes; LastWriteTime 2026-10-30 05:49:03 +05:30. Recorded before scoring.
- Ordinal conversion equivalence: PASS; max abs EF diff 0.0, zero 40/50 flips, exact equality on 124 baseline keys; CPU fp32, 20 VAL videos.
- Both ordinal checkpoint hashes rechecked against the JSON on 2026-10-31 around 02:56 +05:30: match.
- Converted checkpoint LastWriteTime 01:45:36; equivalence JSON LastWriteTime 01:47:28 on 2026-10-31 +05:30. Existing JSON preserved after overwrite refusal.
- First-20-VAL order confirmed; both models in eval() in the saved check script.
- No seed-101 scoring folders found in results/eval_split, results/eval_runs or results_ordinal_readout. Limitation: this is not proof no command was attempted; owner is uncertain from memory.
- Claude review forwarded on 2026-10-31 around 03:03 +05:30 accepted the evidence and cleared planned VAL scoring.
- No new training, scoring or TEST access performed in these verification steps.

## Pair-101 analysis and review - logged 2026-10-31 03:28:48 +05:30
- Analysis executed on 2026-10-31 around 03:23 +05:30 after scoring results were visible. Tag pair_10 maps to seed pair 101; suggested tag was pair_101. One execution observed in chat. Output preserved at results_pair_analysis/pair_10/analysis.json without rerun or rename.
- Frozen analysis settings retained: paired bootstrap, 10000 resamples, seed 404, 95% percentile intervals; populations 1288 all, 133 mild, 999 Normal; metric order mild, Normal, MAE.
- Rounded descriptive differences (ordinal minus baseline): mild recall +0.038, CI [-0.038, +0.113]; Normal recall -0.008, CI [-0.022, +0.005]; MAE +0.158, CI [-0.029, +0.342]. No final across-seed label or guardrail verdict.
- Script ran before Claude review. Post-run review forwarded around 03:26 +05:30 found no error requiring rerun, accepted the recorded analysis and cleared baseline seed 202 under the unchanged plan. Review timing recorded as a process deviation; no statistical setting, threshold, arm or pair was changed for this analysis.
- Current compare_pair.py SHA-256: 110BDBAF663816681411AB913F35F7930E14A2421B7B0487FF53BC2C859E6AB6.
- Current analysis.json SHA-256: 07941F913EA9EDA1CF975B976A506D8CD46A30F463B56B44D9EF5F7525FF5EC8. Hashes captured around 03:28 +05:30; record current contents, not proof of historical execution bytes.
- Secondary four-class head: accuracy 0.812; recalls <40 0.654, 40-50 0.241, 50-70 0.947, >=70 0.000 on 36 videos. Separate from primary regression-threshold results. Merged Normal view pending full confusion matrix.
- No TEST evaluation or training changes. Remaining pairs follow the frozen plan regardless of this pair's result.

- Baseline seed 202 launch attempted: 2026-10-31 03:30:15 +05:30. Owner confirmed plugged in, sleep off and other GPU-heavy apps closed.

## Baseline seed 202 completion - logged 2026-10-31 12:34:47 +05:30
- Matched baseline, seed 202, config configs/base.yaml; frozen training settings retained. Offline run: offline-run-20261031_033022-i4bsjs2a.
- Completion output reported in chat on 2026-10-31 around 12:32 +05:30. Actual finish time unknown.
- Early stopping at epoch 18 of 40 after 6 epochs without improvement. Best epoch 12; best VAL MAE 5.956 (rounded training output). Final epoch VAL MAE 6.05698 is not the best checkpoint's score.
- Train 7465 / VAL 1288 clips. Three initial non-finite-loss warnings shown; no further such warnings shown after epoch 1.
- Confirmed checkpoints/r3d18_matched_baseline_seed202/best.pt: 133007041 bytes, LastWriteTime 2026-10-31 04:27:14 +05:30. Checkpoint save time is not the training finish time.
- Confirmed provenance.json: 2042 bytes, LastWriteTime 2026-10-31 03:30:26 +05:30.
- No standalone scoring or TEST access reported for seed 202. Training included validation. Ordinal seed 202 not yet started in this exchange.

- Ordinal seed 202 launch attempted: 2026-10-31 13:41:44 +05:30. Owner confirmed plugged in, sleep off and other GPU-heavy apps closed.

## Ordinal seed 202 completion - logged 2026-10-31 16:46:54 +05:30
- Ordinal arm, seed 202, config configs/base.yaml; frozen training settings retained. Offline run: offline-run-20261031_134206-x45y3uaw.
- Completion output reported in chat on 2026-10-31 around 16:45 +05:30. Actual finish time unknown.
- Early stopping at epoch 32 of 40 after 6 epochs without improvement. Best epoch 26; best VAL MAE 5.334 (rounded training output). Final epoch 32 VAL MAE 5.40501 is not the best checkpoint's score.
- Train 7465 / VAL 1288 clips. Three initial non-finite-loss warnings shown; no further such warnings shown after epoch 1.
- Confirmed checkpoints/r3d18_ordinal_seed202/best.pt: 133010600 bytes, LastWriteTime 2026-10-31 16:07:31 +05:30. Checkpoint save time is not the training finish time.
- Confirmed provenance.json: 2248 bytes, LastWriteTime 2026-10-31 13:42:10 +05:30.
- No standalone scoring or TEST access reported for ordinal seed 202. Training included validation. Conversion, hash capture and pair-202 scoring remain pending.

## Pair-202 conversion and hashes - logged 2026-10-31 16:50:30 +05:30
- Converter reported 5 allowlisted ordinal-head keys removed and 124 baseline keys kept, with zero missing keys and matching shapes/dtypes. New best_for_eval.pt written; original best.pt unchanged, reported hash prefix 95b9b6625fb3466f.
- The full equivalence check passed once on seed 101 as registered; it was not rerun for seed 202. Seed-202 conversion used the reviewed per-file checks.
- Current baseline seed-202 best.pt SHA-256: 6F9687F3ABF268009292C6DF4B3070AF0F5F0407D7C844B72B3FEC97C199B313.
- Current ordinal seed-202 best.pt SHA-256: 95B9B6625FB3466F1F7A0BB288DD87655943DFEE8E009D74772D4EDCCA74C467.
- Current ordinal seed-202 best_for_eval.pt SHA-256: FE1CCD6606A28B2A09CFCB960C3941AE04BF2ABE787FE25F2FCE4AB54EC7266E.
- Hashes captured around 2026-10-31 16:49 +05:30. No standalone pair-202 scoring or TEST access reported yet.

## Pair-202 analysis - logged 2026-10-31 17:28:32 +05:30
- compare_pair_202.py reviewed by Claude before execution; review forwarded 2026-10-31 around 17:23 +05:30. Analysis output reported around 17:26 +05:30, tag pair_202. One execution observed in chat.
- Script SHA-256: 4F15024652D7489DECBC053D2E7367BC8966404A8991B63069D4AD943BF11CFA. Copied from reviewed compare_pair.py with only four fingerprint constants changed; line comparison showed only the two EXPECTED_ lines changed.
- Old pair-101 wording remains in descriptive text and the example command. Actual inputs were the two seed-202 result folders, with tag pair_202 and matching seed-202 fingerprint constants.
- Frozen settings: paired bootstrap, 10000 resamples, seed 404, 95% percentile intervals; populations 1288 all, 133 mild, 999 Normal; metric order mild, Normal, MAE. Video filenames and labels matched exactly.
- Rounded differences, ordinal minus baseline: mild recall +0.165, CI [+0.075, +0.263]; Normal recall -0.018, CI [-0.030, -0.006]; MAE -0.621, CI [-0.816, -0.436]. Single-pair descriptive results, not the final three-seed verdict.
- Output: results_pair_analysis/pair_202/analysis.json. Current SHA-256: B98763C55967EAE435123E46B10BFCE87CEADC8AEE476CDA731466127F3DB3A0. Preserve the record without rerun or rename.
- Secondary four-class head, ordinal_seed202_val: accuracy 0.816; recalls <40 0.603, 40-50 0.226, 50-70 0.962, >=70 0.028 on 36 videos. Separate from primary EF-threshold results.
- Owner confirmed around 17:24 +05:30 that no seed-303 training, scoring or analysis had started independently. Combined-analysis script preparation and review remain pending before seed 303. No TEST evaluation or training-setting changes.

## Combined analysis implementation frozen - 2026-10-31 18:55:19 +05:30
- Local commit d7a92945e03772bb797317c51d7867d7b0d9b11d contains only compare_combined.py and docs/prereg_addendum_combined.md. Original preregistration unchanged.
- Script working-file SHA-256 (CRLF): 261964CF64761E2CB4A3861DF42F84AE1E44DA668848D77A6E8A48B845A57E0E. Staged LF bytes verified before commit: 031ea16fac4356dd1179008790c4d90aa395a6925921920f40702599ced2b3e5.
- Addendum working-file SHA-256 (CRLF): 2DC7908FAEF1D0335E5C99248523C34B04050E81B7ADF92FB998325ACD2C9584. Staged LF bytes verified before commit: edb7921f0731e8a4069c2fcee0144dce79d07fc58e5ecf86117770fd3e4aee4c.
- core.autocrlf=true; check-attr returned no overrides. Saved script diff matched the reviewed edits. Pre-review backup remains at docs/compare_combined_before_review.py, untracked.
- Four synthetic label cases passed on the actual saved script around 18:41 +05:30: all positive with guards, positive with failed guard, exactly-zero pair, all negative. No real combined analysis or partial two-pair run performed.
- Later combined run must reproduce saved per-pair values at printed 3-decimal precision. Pair 101 (pair_10): mild baseline 0.286, ordinal 0.323, diff +0.038; Normal diff -0.008; MAE diff +0.158. Pair 202: mild baseline 0.195, ordinal 0.361, diff +0.165; Normal diff -0.018; MAE diff -0.621. Any mismatch means stop and inspect, not rerun.
- Owner confirmed around 18:33 +05:30 that no independent seed-303 training, scoring or analysis had started. No seed-303 job has been launched in this exchange.
- Combined analysis remains blocked until all three pairs are scored and only the four pending seed-303 fingerprint constants are filled, with diff and hash recorded and reviewed. No TEST evaluation or training-setting changes.

- Baseline seed 303 launch attempted: 2026-10-31 18:57:59 +05:30. Owner confirmed plugged in, sleep off and no other training or GPU-heavy app running.

## Baseline seed 303 completion - logged 2026-10-31 22:20:38 +05:30
- Matched baseline, seed 303, config configs/base.yaml; frozen training settings retained. Offline run: offline-run-20261031_185805-ydnbd1y5.
- Completion output reported in chat on 2026-10-31 around 22:17 +05:30. Actual finish time unknown.
- Early stopping at epoch 31 of 40 after 6 epochs without improvement. Best epoch 25; best VAL MAE 5.315 (rounded training output). Final epoch VAL MAE 5.55757 is not the best checkpoint's score.
- Train 7465 / VAL 1288 clips. Two initial non-finite-loss warnings shown; no further such warnings shown after epoch 1.
- Confirmed checkpoints/r3d18_matched_baseline_seed303/best.pt: 133007041 bytes, LastWriteTime 2026-10-31 21:20:46 +05:30. Checkpoint save time is not the training finish time.
- Confirmed provenance.json: 2042 bytes, LastWriteTime 2026-10-31 18:58:08 +05:30.
- No standalone scoring or TEST access reported for seed 303. Training included validation. Ordinal seed 303 not yet started in this exchange.

- Ordinal seed 303 launch attempted: 2026-10-31 22:23:38 +05:30. Owner confirmed plugged in, sleep off and other GPU-heavy apps closed.

## Ordinal seed 303 completion - logged 2026-11-01 02:00:07 +05:30
- Ordinal arm, seed 303, config configs/base.yaml; frozen training settings retained. Offline run: offline-run-20261031_222343-butle2u3.
- Completion output reported in chat 2026-11-01 around 01:58 +05:30; actual finish time unknown.
- Early stopping at epoch 31 of 40 after 6 epochs without improvement. Best epoch 25; best VAL MAE 5.425 (rounded training output). Final epoch 31 VAL MAE 5.67784 is not the best checkpoint's score.
- Train 7465 / VAL 1288 clips. Two initial non-finite-loss warnings shown at start (trainer labels them warmup); printed epoch metrics are finite afterward.
- Confirmed checkpoints/r3d18_ordinal_seed303/best.pt: 133010600 bytes, LastWriteTime 2026-11-01 00:45:48 +05:30. Checkpoint save time is not the training finish time.
- Confirmed provenance.json: 2248 bytes, LastWriteTime 2026-10-31 22:23:47 +05:30.
- All six training runs complete. No standalone seed-303 scoring or TEST access reported. Training included validation. Conversion, hash capture and pair-303 scoring remain pending.

## Pair-303 conversion and hash record - logged 2026-11-01 02:03:35 +05:30
- Conversion reported 2026-11-01 around 02:01 +05:30: strip_ordinal_keys.py checkpoints/r3d18_ordinal_seed303/best.pt -> checkpoints/r3d18_ordinal_seed303/best_for_eval.pt.
- Removed 5 allowlisted ordinal_head keys; kept 124 baseline keys verified against current R3DMultiTask (zero missing, shapes and dtypes equal). New file written; original unchanged, printed source SHA-256 prefix f527aaf8db53bfb0.
- Full two-level equivalence gate passed once on seed 101; not repeated per seed under the frozen plan.
- Baseline303 best.pt SHA-256: 168E7D732C5DC83D4B77204A61B8559D58693CEC1D9109EF51535A407E250F25
- Ordinal303 best.pt SHA-256: F527AAF8DB53BFB08420BC0C755554A3493B4E851F4E96A09C863B42E5BD900B
- Ordinal303 best_for_eval.pt SHA-256: 5B5532282067A41BBD0F7B048F3F4CCB7F9797ADF116011ABF92BB519AB2DF5F
- Hashes describe current file contents only. No standalone scoring or TEST access reported for pair 303 yet.

## Pair-303 analysis review and pre-run record - logged 2026-11-01 02:25:48 +05:30
- compare_pair_303.py derived from reviewed compare_pair_202.py by four fingerprint substitutions. Source SHA-256 at creation: 4F15024652D7489DECBC053D2E7367BC8966404A8991B63069D4AD943BF11CFA. Laptop line diff showed only the two EXPECTED_ lines replaced. No statistical logic intentionally changed.
- New script SHA-256 rechecked now: 6ED6F03E39584F9D7B0DD724BB62834E1E2EF0F61C6A6E613C76180A0685B4A2. Claude review forwarded by user around 2026-11-01 02:24 +05:30 accepted as-is. Cosmetic pair-101 wording retained. Reviewer's abbreviated 6ED6F039 hash was a typo; full verified hash above governs.
- Fingerprint chronology: checkpoint full hashes captured 2026-11-01 02:01:59 +05:30, before standalone scoring. Prediction full hashes captured 02:15:56, after scoring. All four script constants filled after scoring and reviewed before paired analysis. These guards lock observed file contents; they do not independently prove earlier training history.
- Pair-303 summaries were seen before paired analysis: baseline/ordinal mild 38/133 vs 31/133; Normal 954/999 vs 950/999; MAE 5.315 vs 5.425. Bootstrap method was registered earlier. Combined code and interpretation addendum were frozen before the controlled 303 runs; prior post-101/202 clarifications remain disclosed. No formal paired intervals or combined verdict yet.
- Earlier no-independent-303-work confirmation at 2026-10-31 around 18:33 described that time only. Controlled baseline303 launch followed readiness at 18:56; ordinal303 launch followed readiness around 22:21. Both runs and scoring are now complete; the earlier confirmation is not a current no-work claim.
- Future combined PENDING-303 fill must use these same four prefixes, with diff review and new script hash recorded before execution, as required by the addendum.
- results_pair_analysis/pair_303 absent at this check. This block only records review; it does not execute the analysis.

## Pair-303 paired analysis - logged 2026-11-01 02:29:03 +05:30
- Completion output reported 2026-11-01 around 02:27 +05:30 for compare_pair_303.py with baseline val_matched_baseline303_fp32_k1, ordinal val_ordinal303_stripped_fp32_k1, tag pair_303. Reviewed script SHA-256: 6ED6F03E39584F9D7B0DD724BB62834E1E2EF0F61C6A6E613C76180A0685B4A2; review recorded before execution.
- Alignment PASS: 1288 videos, identical filenames and labels in the same order. Mild n=133, Normal n=999. Frozen bootstrap: 10000 resamples, seed 404, 95% percentile intervals.
- Mild recall: baseline 0.286, ordinal 0.233, diff -0.053 (95% CI -0.128 to +0.023).
- Normal recall: baseline 0.955, ordinal 0.951, diff -0.004 (95% CI -0.014 to +0.005).
- MAE: baseline 5.315, ordinal 5.425, diff +0.110 (95% CI -0.038 to +0.254). All differences ordinal minus baseline; descriptive anchors match.
- Output results_pair_analysis/pair_303/analysis.json SHA-256 captured at 02:28: 28420949BD88B4EFD326AD779FA8B8D31F3B5D2AAB589659E2E0D4FAD791B788.
- Single-pair results are descriptive, with no final label or guardrail verdict. Combined three-pair analysis still pending constant fill, review and single run. No TEST access reported.

## Combined analysis fill review and pre-run record - logged 2026-11-01 02:38:52 +05:30
- Four seed-303 fingerprints filled around 2026-11-01 02:30 +05:30: checkpoint baseline 168e7d732c5dc83d, ordinal 5b5532282067a41b; predictions baseline 69a912f31c7ff551, ordinal f5df0a08d5d50cdf. Same values as reviewed pair-303 script. Pre-fill working SHA-256 verified: 261964CF64761E2CB4A3861DF42F84AE1E44DA668848D77A6E8A48B845A57E0E.
- Filled script SHA-256 rechecked now: 87FFEAFE12FDA19562683C7AEE74D17644801488C8DA92AB06424821A648C840. This on-disk CRLF hash is the expected script_sha256 in the output, not the frozen Git LF hash.
- Fill diff showed only two constant lines replaced plus a blank display marker. Read-only reverse-fill check passed around 02:32: reversing exactly the four fills restored frozen commit d7a92945e03772bb797317c51d7867d7b0d9b11d byte-for-byte after CRLF conversion. Frozen LF SHA-256 031ea16fac4356dd1179008790c4d90aa395a6925921920f40702599ced2b3e5. No other content change or added/removed blank line.
- Claude acceptance user-forwarded around 02:35; requested seed-101 cross-check completed around 02:38. All four 101 prefixes agree across combined script, saved pair_10 analysis inputs and both metrics.json files. Actual saved tag is pair_10, not pair_101. PENDING-303 remains only as safety-check text, not constant values.
- Chronology: 303 checkpoint hashes captured before standalone scoring; prediction hashes captured afterward; constants filled after seeing all three pair results. Earlier no-independent-303-work confirmation described the pre-run state only and is superseded by the controlled runs. Method and interpretation freeze chronology, including post-101/202 clarifications, remains as previously logged; nothing changed for these results.
- Post-run anchors at 3 decimals (ordinal minus baseline): 101 mild +0.038, Normal -0.008, MAE +0.158; 202 mild +0.165, Normal -0.018, MAE -0.621; 303 mild -0.053, Normal -0.004, MAE +0.110. Any mismatch means stop and inspect, no rerun.
- Combined output directory absent at this check. Single combined run not executed by this block. Frozen addendum label rules apply; no final verdict yet. No TEST access reported.

## Combined three-pair result - logged 2026-11-01 02:42:50 +05:30
- Single combined run completion reported 2026-11-01 around 02:40 +05:30. Saved JSON script_sha256 matches reviewed CRLF file: 87FFEAFE12FDA19562683C7AEE74D17644801488C8DA92AB06424821A648C840.
- Alignment PASS across all six runs: 1288 videos, identical filenames and labels in order. Cluster bootstrap over videos retaining seeds together: 10000 resamples, seed 404, 95% percentile intervals. Populations mild 133, Normal 999, MAE 1288.
- Mild recall mean: baseline 0.2556390977, ordinal 0.3057644110, diff +0.0501253133, CI [0.0075187970, 0.0977443609]. Normal recall mean: baseline 0.9519519520, ordinal 0.9419419419, diff -0.0100100100, CI [-0.0170170170, -0.0033366700].
- MAE mean: baseline 5.5339146670, ordinal 5.4161604307, diff -0.1177542363, CI [-0.2181897435, -0.0173403974]. This averages absolute errors across seeds, not predictions; it is not an ensemble score.
- All per-pair anchors matched saved results at 3 decimals: 101 (saved tag pair_10) mild +0.038, Normal -0.008, MAE +0.158; 202 +0.165, -0.018, -0.621; 303 -0.053, -0.004, +0.110.
- Both guardrails hold: ordinal Normal mean 0.9419419419 >= 0.92 (margin +0.0219419419); MAE diff -0.1177542363 <= +0.15 (margin +0.2677542363).
- Frozen label: inconclusive. Mild differences have mixed signs across seeds; the positive combined interval does not override the required cross-seed consistency. Intervals cover video-sample uncertainty for these trained models, not full training randomness.
- Output results_pair_analysis/combined/combined_analysis.json SHA-256 captured around 02:41: 66B191932D2F7AF291F8C9A0446725BB7125968718DB4DD6DEF721189CC6C34C. Saved JSON inspected and agrees with terminal output, input fingerprints and reviewed script hash. Six training runs plus planned scoring, paired and combined analyses complete. No new TEST access reported; prior exposure and provenance limitations remain disclosed.
