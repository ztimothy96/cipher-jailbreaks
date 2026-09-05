"""Converts data/polyrefuse/*.json (see data/polyrefuse/README.md) into the
CSV format probe_generalization/dataset.py expects.

Usage:
    python3 probe_generalization/prepare_polyrefuse_dataset.py
"""

import json
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent.parent / "data" / "polyrefuse"
OUT_DIR = SOURCE_DIR


def convert(source_name: str, out_name: str, column: str):
    with open(SOURCE_DIR / source_name) as f:
        records = json.load(f)
    df = pd.DataFrame({column: [r["instruction"] for r in records]})
    out_path = OUT_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def main():
    convert("harmful_test_translated_en.json", "harmful.csv", "goal")
    convert("harmless_test_translated_en.json", "harmless.csv", "prompt")


if __name__ == "__main__":
    main()
