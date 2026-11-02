"""Count videos per EF group for TRAIN and VAL only (pre-registered spec).

NEW FILE. Streams FileList.csv row by row. For each row it reads ONLY the
Split value first; the EF value is parsed and counted only for TRAIN and VAL
rows. TEST rows are skipped without their labels being inspected, counted, or
stored (the file itself is shared, so TEST lines pass through the CSV parser,
but nothing is read out of them beyond the Split field). No files are written.

Guards:
  - EF must parse as a finite number; bad TRAIN/VAL rows are reported.
  - Split totals are checked against the sizes recorded in the original W&B
    run log (TRAIN 7465, VAL 1288); a mismatch prints a warning.

Groups use >= at each cut-off (same convention as the dataset class):
  <40, 40-50 (40 <= EF < 50), 50-70 (50 <= EF < 70), >=70.

Run from echo-ef-project/ (PowerShell):
    python count_ef_groups.py
or with an explicit path:
    python count_ef_groups.py "data\EchoNet-Dynamic\FileList.csv"
"""
import csv
import math
import sys

NAMES = ["<40", "40-50", "50-70", ">=70"]
EXPLORATORY_MIN_VAL = 30  # pre-set rule: VAL groups below this are exploratory
EXPECTED_TOTALS = {"TRAIN": 7465, "VAL": 1288}  # recorded in the original W&B run log


def group_index(ef: float) -> int:
    if ef < 40:
        return 0
    if ef < 50:
        return 1
    if ef < 70:
        return 2
    return 3


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else r"data\EchoNet-Dynamic\FileList.csv"
    counts = {"TRAIN": [0, 0, 0, 0], "VAL": [0, 0, 0, 0]}
    bad_rows = 0
    other_split = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("Split") or "").strip().upper()
            if split == "TEST":
                continue  # skip: label not inspected, counted, or stored
            if split not in counts:
                other_split += 1
                continue
            try:
                ef = float(row["EF"])
            except (TypeError, ValueError):
                bad_rows += 1
                continue
            if not math.isfinite(ef):
                bad_rows += 1
                continue
            counts[split][group_index(ef)] += 1

    for split in ("TRAIN", "VAL"):
        c = counts[split]
        total = sum(c)
        line = ", ".join(f"{n}={v}" for n, v in zip(NAMES, c))
        print(f"{split}: n={total} | {line}")
        expected = EXPECTED_TOTALS[split]
        if total != expected:
            print(f"  WARNING: {split} total {total} differs from the recorded {expected}; "
                  f"check that this FileList.csv is the same version the runs used")
        if split == "VAL":
            for n, v in zip(NAMES, c):
                if v < EXPLORATORY_MIN_VAL:
                    print(f"  FLAG: VAL group {n} has {v} videos (< {EXPLORATORY_MIN_VAL}) "
                          f"-> report as exploratory")
    if bad_rows:
        print(f"WARNING: {bad_rows} TRAIN/VAL rows had a missing or non-finite EF and were not counted")
    if other_split:
        print(f"Note: {other_split} rows had an unrecognized Split value and were not counted")
    print("TEST rows were skipped without their labels being inspected, counted, or stored.")


if __name__ == "__main__":
    main()

