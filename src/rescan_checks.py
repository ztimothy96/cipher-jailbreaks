"""Recomputes understanding_ok on an existing understanding_check
output file, from its stored `completion` field — no model call, no Modal.

Use this whenever only local scoring logic (looks_like_noise) changes, not
the actual model responses. Re-running understanding_check.py itself
re-generates from the model, which costs real GPU time for a fix that has
nothing to do with generation.

Usage:
    python3 src/rescan_checks.py \\
        --in results/raw/understanding_check__Qwen__Qwen2.5-7B-Instruct.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.prompt_rendering import looks_like_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out",
                        default=None,
                        help="defaults to overwriting --in")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out) if args.out else in_path

    records = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["understanding_ok"] = not looks_like_noise(
                record["completion"])
            records.append(record)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Rescanned {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
