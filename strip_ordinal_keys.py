"""Strip the ordinal_head.* keys from an ORDINAL-arm checkpoint so the UNCHANGED
evaluator (src/eval/evaluate_split.py) can load it. NEW FILE; the evaluator is
not modified.

Why: an ordinal checkpoint's model_state_dict is the baseline's plus exactly
five ordinal_head.* keys. The evaluator calls load_state_dict with strict=True
and would stop on those unexpected keys. This tool writes a NEW checkpoint file
with only the allowlisted ordinal keys removed; every other top-level field
(config, epoch, val_mae) is preserved unchanged, so the evaluator's checkpoint
settings comparison and normalisation resolution behave exactly as they do for
a baseline checkpoint from the same trainer.

Safety rules:
  * the input checkpoint is only read (sha256 recorded before and after); it is
    never modified, and the output is a different, not-yet-existing file
  * torch.load uses weights_only=True only; there is no unsafe fallback
  * the retained keys must match the CURRENT baseline R3DMultiTask state dict
    exactly (zero missing, and shapes/dtypes equal); the removed keys must be
    within the five-key allowlist; anything else stops with an error
  * this tool does not evaluate anything and does not touch data splits

Run from the project root, only once a real ordinal checkpoint exists (after an
explicitly approved training run):
    python strip_ordinal_keys.py checkpoints\r3d18_ordinal_seed101\best.pt checkpoints\r3d18_ordinal_seed101\best_for_eval.pt
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.models.r3d_multitask import R3DMultiTask

ORDINAL_KEY_ALLOWLIST = frozenset({
    "ordinal_head.shared.weight",
    "ordinal_head.shared.bias",
    "ordinal_head.b0",
    "ordinal_head.d1",
    "ordinal_head.d2",
})


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_ordinal_checkpoint(input_path, output_path) -> dict:
    """Write a baseline-loadable copy of an ordinal checkpoint. Returns the
    provenance record. Stops with SystemExit on any failed check."""
    in_p, out_p = Path(input_path), Path(output_path)
    if not in_p.is_file():
        raise SystemExit(f"STOPPED: input checkpoint not found: {in_p}")
    if in_p.resolve() == out_p.resolve():
        raise SystemExit("STOPPED: output must be a different file; the original is never overwritten")
    if out_p.exists():
        raise SystemExit(f"STOPPED: {out_p} already exists; refusing to overwrite")

    input_hash = _sha256_of(in_p)
    try:
        ckpt = torch.load(in_p, map_location="cpu", weights_only=True)
    except Exception as e:
        raise SystemExit(f"STOPPED: torch.load(weights_only=True) failed for {in_p}:\n"
                         f"{type(e).__name__}: {e}\nNo unsafe fallback is provided.")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise SystemExit("STOPPED: checkpoint is not a dict with a 'model_state_dict' dict")
    sd = ckpt["model_state_dict"]

    baseline_sd = R3DMultiTask(pretrained=False).state_dict()  # key/shape reference
    baseline_keys = set(baseline_sd)
    extra = sorted(set(sd) - baseline_keys)
    missing = sorted(baseline_keys - set(sd))
    if missing:
        raise SystemExit(f"STOPPED: {len(missing)} baseline keys are missing (first: {missing[0]}); "
                         f"this does not match the current baseline architecture")
    if not extra:
        raise SystemExit("STOPPED: no ordinal_head.* keys found; this looks like a baseline "
                         "checkpoint - evaluate it directly, no stripping needed")
    not_allowed = [k for k in extra if k not in ORDINAL_KEY_ALLOWLIST]
    if not_allowed:
        raise SystemExit(f"STOPPED: unexpected keys outside the ordinal allowlist: {not_allowed}; "
                         f"refusing to strip anything not on the allowlist")
    for k in baseline_keys:
        if not (sd[k].shape == baseline_sd[k].shape and sd[k].dtype == baseline_sd[k].dtype):
            raise SystemExit(f"STOPPED: shape/dtype mismatch on '{k}': checkpoint "
                             f"{tuple(sd[k].shape)}/{sd[k].dtype} vs baseline "
                             f"{tuple(baseline_sd[k].shape)}/{baseline_sd[k].dtype}; "
                             f"the architecture may have changed - re-review before proceeding")

    provenance = {
        "source_file": str(in_p),
        "source_sha256": input_hash,
        "removed_keys": extra,
        "stripped_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "ordinal_head.* keys removed so the unchanged evaluator can load this file",
    }
    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = {k: v for k, v in sd.items() if k not in ORDINAL_KEY_ALLOWLIST}
    new_ckpt["ordinal_strip_provenance"] = provenance
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out_p)

    if _sha256_of(in_p) != input_hash:
        raise SystemExit("STOPPED: input file changed during processing; do not use the output")
    print(f"Removed {len(extra)} allowlisted keys: {extra}")
    print(f"Kept {len(new_ckpt['model_state_dict'])} baseline keys (verified against the current "
          f"R3DMultiTask: zero missing, shapes and dtypes equal)")
    print(f"Wrote {out_p} (new file; the original {in_p} is unchanged, sha256 {input_hash[:16]}...)")
    return provenance


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python strip_ordinal_keys.py <ordinal_checkpoint> <new_output_checkpoint>")
    strip_ordinal_checkpoint(sys.argv[1], sys.argv[2])
