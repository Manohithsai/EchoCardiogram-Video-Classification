"""
Read-only diagnostic for the two saved Swin VAL output folders.

It reports raw file hashes, text encoding/newline details, metadata, and whether
video_ids.txt exactly matches the Unicode file_names array in predictions.npz.
It does not load checkpoints or models, does not run inference, and writes nothing.

Run from echo-ef-project:

    python diagnose_swin_video_ids.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FOLDER_NAMES = (
    "val_swin_fp32_k1",
    "val_swin3d_t_baseline_fp32_k1",
)
META_FIELDS = (
    "tag",
    "split",
    "n",
    "precision",
    "autocast_active",
    "clips_per_video",
    "checkpoint_sha256_16",
    "predictions_sha256_16",
    "video_ids_sha256_16",
    "normalisation_source",
    "eval_config_sha256_16",
    "file_list_csv_sha256_16",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(data):
    return sha256_bytes(data)[:16]


def first_byte_mismatch(a, b):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None


def first_line_mismatch(a, b):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return limit, None, None
    return None


def inspect_folder(results_root, folder_name):
    run_dir = results_root / folder_name
    print(f"=== {folder_name} ===")
    if not run_dir.is_dir():
        print("missing folder")
        print("")
        return None

    metrics_path = run_dir / "metrics.json"
    pred_path = run_dir / "predictions.npz"
    ids_path = run_dir / "video_ids.txt"
    for path in (metrics_path, pred_path, ids_path):
        print(f"{path.name}: exists={path.is_file()}")
    if not all(path.is_file() for path in (metrics_path, pred_path, ids_path)):
        print("")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("metadata:")
    for field in META_FIELDS:
        print(f"  {field}: {meta.get(field)!r}")

    pred_hash = sha256_file(pred_path)
    ids_bytes = ids_path.read_bytes()
    ids_hash = sha256_bytes(ids_bytes)
    print("raw hashes:")
    print(f"  predictions.npz full: {pred_hash}")
    print(f"  predictions.npz[:16]: {pred_hash[:16]}")
    print(f"  video_ids.txt full: {ids_hash}")
    print(f"  video_ids.txt[:16]: {ids_hash[:16]}")
    print(f"  metrics recorded video_ids_sha256_16: {meta.get('video_ids_sha256_16')}")
    print(f"  video_ids raw hash matches metadata: {ids_hash[:16] == meta.get('video_ids_sha256_16')}")

    crlf_count = ids_bytes.count(b"\r\n")
    lf_count = ids_bytes.count(b"\n")
    cr_only_count = ids_bytes.count(b"\r") - crlf_count
    print("text bytes:")
    print(f"  size: {len(ids_bytes)} bytes")
    has_bom = ids_bytes.startswith(bytes([0xEF, 0xBB, 0xBF]))
    print(f"  UTF-8 BOM: {has_bom}")
    print(f"  CRLF endings: {crlf_count}; LF bytes: {lf_count}; CR-only bytes: {cr_only_count}")
    print(f"  first 80 bytes: {ids_bytes[:80]!r}")
    print(f"  last 80 bytes: {ids_bytes[-80:]!r}")

    try:
        text = ids_bytes.decode("utf-8")
        text_lines = text.splitlines()
        print(f"  UTF-8 decode: ok; splitlines count={len(text_lines)}; unique lines={len(set(text_lines))}")
        print(f"  first 3 text lines: {text_lines[:3]!r}")
        print(f"  last 3 text lines: {text_lines[-3:]!r}")
    except UnicodeDecodeError as e:
        text_lines = None
        print(f"  UTF-8 decode: FAILED: {e}")

    with np.load(pred_path, allow_pickle=False) as z:
        names_arr = z["file_names"]
        names = [str(x) for x in names_arr.tolist()]
    print("NPZ file_names:")
    print(f"  shape={names_arr.shape}; dtype={names_arr.dtype}; unique={len(set(names))}")
    print(f"  first 3: {names[:3]!r}")
    print(f"  last 3: {names[-3:]!r}")

    canonical_lf = ("\n".join(names) + "\n").encode("utf-8")
    canonical_crlf = ("\r\n".join(names) + "\r\n").encode("utf-8")
    print("canonical reconstructions from NPZ names:")
    print(f"  LF bytes hash[:16]={short_hash(canonical_lf)}; exact match={ids_bytes == canonical_lf}")
    print(f"  CRLF bytes hash[:16]={short_hash(canonical_crlf)}; exact match={ids_bytes == canonical_crlf}")
    for label, canonical in (("LF", canonical_lf), ("CRLF", canonical_crlf)):
        pos = first_byte_mismatch(ids_bytes, canonical)
        if pos is None:
            print(f"  first byte difference vs {label}: none")
        else:
            print(f"  first byte difference vs {label}: offset {pos}; actual={ids_bytes[pos:pos+30]!r}; expected={canonical[pos:pos+30]!r}")

    if text_lines is not None:
        print(f"text lines exactly match NPZ names: {text_lines == names}")
        mismatch = first_line_mismatch(text_lines, names)
        if mismatch is None:
            print("  first line mismatch: none")
        else:
            i, actual, expected = mismatch
            print(f"  first line mismatch at index {i}: actual={actual!r}; expected={expected!r}")
            print(f"  line counts actual={len(text_lines)}; NPZ={len(names)}")
    print("")
    return {
        "folder": folder_name,
        "meta": meta,
        "pred_hash": pred_hash,
        "ids_hash": ids_hash,
        "ids_bytes": ids_bytes,
        "names": names,
        "text_lines": text_lines,
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only Swin video_ids diagnostic")
    parser.add_argument("--results-root", default="results/eval_split")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    results = [inspect_folder(results_root, name) for name in FOLDER_NAMES]
    present = [r for r in results if r is not None]
    if len(present) == 2:
        a, b = present
        print("=== cross-folder comparison ===")
        print(f"predictions.npz raw bytes/hash identical: {a['pred_hash'] == b['pred_hash']}")
        print(f"video_ids.txt raw bytes/hash identical: {a['ids_hash'] == b['ids_hash']}")
        print(f"NPZ file_names identical: {a['names'] == b['names']}")
        if a["text_lines"] is not None and b["text_lines"] is not None:
            print(f"text lines identical: {a['text_lines'] == b['text_lines']}")
        for field in META_FIELDS:
            same = a["meta"].get(field) == b["meta"].get(field)
            print(f"metadata {field}: same={same}; {a['meta'].get(field)!r} vs {b['meta'].get(field)!r}")
    print("Diagnostic complete. No files were written or changed.")


if __name__ == "__main__":
    main()
