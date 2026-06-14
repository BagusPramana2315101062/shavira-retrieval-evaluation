"""Cek cepat struktur dataset SHAVIRA.

Contoh:
    python src/check_dataset.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shavira_retrieval.constants import DEFAULT_JSONL_FILES, DEFAULT_VALIDATION_FILE
from shavira_retrieval.utils import make_hash, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--jsonl-files", nargs="+", default=DEFAULT_JSONL_FILES)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    record_hashes = set()

    for name in args.jsonl_files:
        path = data_dir / name
        print(f"\n[JSONL] {path}")
        if not path.exists():
            print("  TIDAK DITEMUKAN")
            continue

        total = 0
        has_text = 0
        has_metadata = 0
        for obj in read_jsonl(path):
            total += 1
            text = str(obj.get("text", "") or "")
            if text.strip():
                has_text += 1
                record_hashes.add(make_hash(text))
            if isinstance(obj.get("metadata", None), dict):
                has_metadata += 1

        print(f"  total records : {total}")
        print(f"  memiliki text : {has_text}")
        print(f"  memiliki metadata : {has_metadata}")

    val_path = data_dir / args.validation_file
    print(f"\n[VALIDATION] {val_path}")
    if not val_path.exists():
        print("  TIDAK DITEMUKAN")
        return

    df = pd.read_excel(val_path)
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    required = {"ID", "Context", "Question", "Answer"}
    missing = required - set(df.columns)
    print(f"  missing required columns: {sorted(missing)}")

    if "Context" in df.columns:
        context_hashes = df["Context"].astype(str).map(make_hash)
        matches = context_hashes.isin(record_hashes).sum()
        print(f"  Context exact record match: {matches}/{len(df)}")


if __name__ == "__main__":
    main()
