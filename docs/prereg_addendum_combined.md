# Addendum to the ordinal-experiment pre-registration: combined-analysis clarifications

First drafted 2026-10-31 about 17:40 +05:30; final revision 2026-10-31 about 18:28 +05:30.

This addendum records clarifications to section 5 of docs/ordinal_experiment_prereg.md
(final saved version, SHA-256 6A90D9ECCC648C64429B562BE08D4A5B59BEC7C6816FD0562C2940BA1B76481C,
git commit 5da82cf, saved and hashed by the owner at about 02:17 +05:30 on 2026-10-30).
The original pre-registration file is unchanged; nothing here revises it.

Postdating statement: every clarification in this addendum was written and accepted
on 2026-10-31 with only the pair-101 and pair-202 results visible. No seed-303
training, scoring or analysis was started in our workflow, and the owner confirmed
at about 17:24 +05:30 on 2026-10-31 that none had been started independently; a
fresh confirmation will be obtained before the combined analysis is frozen if the
delay makes that stale. This addendum settles implementation details of the
combined analysis before any seed-303 results exist.

## What section 5 already froze

Section 5 of the final pre-registration already contains the bootstrap details, in a
block that sits inside section 5 after both the per-pair and the combined-estimate
paragraphs. Literal text of the saved file (read back from the owner's laptop on
2026-10-31 at about 18:28 +05:30):

    [review 30 Oct] Bootstrap details, fixed now:
    - Number of resamples: 10,000.
    - Interval: 95% percentile interval.
    - Bootstrap RNG seed: one fixed value, recorded now: 404.
    - Resampling population per metric: mild recall resamples the 133 true-mild VAL
      videos, Normal recall resamples the 963 + 36 true-Normal VAL videos (the >=50
      group, 999 videos), and MAE resamples all 1,288 VAL videos.
    - Effect size: the report states the size of each difference, not only its
      direction and interval, because "consistently positive" does not require a
      minimum effect.

These details sit in section 5 generally and govern the combined cluster bootstrap
as well as the per-pair paired bootstrap. An earlier working note treated them as
unspecified for the combined estimate; that reading was wrong and is corrected here.
The pre-registration hash and commit cited in the header
(6A90D9ECCC648C64429B562BE08D4A5B59BEC7C6816FD0562C2940BA1B76481C, commit 5da82cf)
are of the version that already includes these 30 October additions.

## Clarifications accepted 2026-10-31

1. MAE guardrail threshold. "No more than about 0.15 worse" is pinned to exactly
   0.15 EF points: the guardrail holds if the combined mean ordinal-minus-baseline
   MAE difference is at most +0.15. Disclosure: this value was pinned while the
   pair-101 MAE difference (+0.158) and the pair-202 MAE difference (-0.621) were
   already visible; those were the only pair results in existence. The threshold
   must not be revised after seed-303 results exist. The mean of the per-seed MAE
   differences equals the combined MAE difference, so the script reports one
   number, not two.

2. Normal guardrail. The ordinal arm's mean Normal recall across seeds must be at
   least 0.92, read as an absolute floor.

3. Sign rule for seed pairs. A per-pair primary (mild-recall) difference of exactly
   zero is neither positive nor negative. A result with such a pair is not
   "consistently positive" and not a "consistent drop"; it falls to "inconclusive"
   unless a guardrail fails.

4. Combined MAE construction. Average each video's absolute error across the seeds
   within each arm, then take the per-video ordinal-minus-baseline difference.
   Predictions are not averaged into an ensemble and re-scored.

5. Seed-303 fingerprints. The combined script ships with the seed-303 checkpoint and
   prediction hash constants as PENDING-303 placeholders and refuses to run while any
   placeholder remains. After seed-303 scoring, only those four constants are filled,
   with the edit diff and the resulting file hash recorded and reviewed, and the
   analysis is then run once.

6. Label precedence. The frozen label definitions overlap when the seed pairs are
   mixed or positive while a guardrail fails. Resolved as: a failed guardrail makes
   the result "negative", even when the mild differences are positive in every pair
   or the combined interval includes zero - the conservative reading of the frozen
   "a consistent drop or a failed guardrail". The script prints the label components
   (per-pair signs, interval bounds, guardrail values) next to the label so the
   precedence never hides how close the result was.

## Status at the time of writing

- Pair 101: fully closed (training, conversion gate, scoring, per-pair analysis).
- Pair 202: fully closed (training, conversion, scoring, per-pair analysis logged
  2026-10-31 17:28:32 +05:30).
- Seed 303: not started. The owner confirmed at about 17:24 +05:30 on 2026-10-31
  that no seed-303 training, scoring or analysis had been started independently.
  A fresh confirmation will be obtained before the combined analysis is frozen if
  the delay makes this stale.
- The combined script (compare_combined.py) has been run only on synthetic test
  data at the registered shape (1,288 videos / 133 mild / 999 Normal), never on
  the real results. No partial two-pair combined analysis has been run.
- No TEST evaluation and no training-setting changes are proposed here or anywhere
  in the combined analysis.
