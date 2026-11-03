"""
Export the full per-epoch history of EVERY wandb run in the echo-ef project
to CSV files.

Run from echo-ef-project/ with the venv active:
    python export_wandb_history.py

Creates results\wandb_export\<run name>__<run id>.csv for each run, plus
results\wandb_export\all_runs_summary.csv (final summary metrics per run).
"""
from pathlib import Path

import pandas as pd
import wandb

ENTITY_PROJECT = "manohit1706-bennett-university/echo-ef"
out = Path("results") / "wandb_export"
out.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
summary_rows = []
for run in api.runs(ENTITY_PROJECT):
    hist = run.history(samples=100000, pandas=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run.name)
    hist.to_csv(out / f"{safe_name}__{run.id}.csv", index=False)
    row = {"run_name": run.name, "run_id": run.id, "state": run.state, "created_at": run.created_at,
           "n_epochs_logged": len(hist)}
    row.update({k: v for k, v in run.summary._json_dict.items() if not k.startswith("_") and not isinstance(v, dict)})
    summary_rows.append(row)
    print(f"exported {run.name} ({run.id}): {len(hist)} rows")

pd.DataFrame(summary_rows).to_csv(out / "all_runs_summary.csv", index=False)
print(f"\nDone. Files in {out.resolve()}")
