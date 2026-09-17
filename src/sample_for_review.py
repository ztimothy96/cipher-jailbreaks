"""Sample translations for manual review, per language, as a Markdown file
that's easy to read and annotate directly in an editor.

Why this is the *primary* QA mechanism: an earlier automated round-trip 
similarity check (LaBSE sentence embeddings) caught zero of the real translation
errors found by manual review of a smoke test. Back-translation kept silently
"fixing" the defect it was supposed to help catch, so it was removed.

Usage:
    python3 src/sample_for_review.py \\
        --translations results/src/translations.jsonl \\
        --sample-size 30 --seed 0
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--translations",
        default="results/src/translations.jsonl")
    parser.add_argument("--out-dir",
                        default="results/src/review")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Prompts to sample per language, independent of dataset size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed seed so re-running produces the same sample "
        "(add --seed N to draw a different one if you want more coverage).")
    args = parser.parse_args()

    by_language = defaultdict(list)
    with open(args.translations) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_language[record["language"]].append(record)

    if not by_language:
        raise SystemExit(f"No records found in {args.translations}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for language, records in sorted(by_language.items()):
        rng = random.Random(args.seed)
        sample = rng.sample(records, min(args.sample_size, len(records)))
        sample.sort(key=lambda r: (r["prompt_id"], r["label"]))

        out_path = out_dir / f"{language}.md"
        with open(out_path, "w") as f:
            f.write(f"# Manual review sample: {language}\n\n")
            f.write(
                f"{len(sample)} of {len(records)} translations, seed={args.seed}. "
                "Fill in a Verdict for each (correct / minor / major) and any "
                "notes, then keep this file — it's the calibration record for "
                "this language's translation accuracy.\n\n")
            for r in sample:
                unk_flag = " (has <unk> token)" if r.get(
                    "has_unk_token") else ""
                f.write(f"## prompt_id={r['prompt_id']} "
                        f"label={r['label']}{unk_flag}\n\n")
                f.write(f"- **Original**: {r['original']}\n")
                f.write(f"- **Translated**: {r['translated']}\n")
                f.write("- **Verdict**: \n")
                f.write("- **Notes**: \n\n")

        print(f"[{language}] wrote {len(sample)}-prompt sample to {out_path}")


if __name__ == "__main__":
    main()
