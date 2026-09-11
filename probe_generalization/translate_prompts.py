"""Translate the probe dataset into Chinese/Japanese/Spanish via the DeepL
API.

QA is manual review of a random sample (probe_generalization/
sample_for_review.py).

Setup: see probe_generalization/deepl_translate.py (DEEPL_API_KEY / .env).

Usage:
    python3 probe_generalization/translate_prompts.py --smoke-test
    python3 probe_generalization/translate_prompts.py \\
        --harmful-csv path/to/advbench.csv --harmless-csv path/to/harmless.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_generalization.shared.dataset import load_probe_dataset, load_smoke_test_dataset
from probe_generalization.shared.deepl_translate import batch_translate, get_translator
from probe_generalization.shared.formats import LANGUAGE_CODES, TEST_LANGUAGES


def load_completed(out_path: Path) -> set[tuple[int, int, str]]:
    if not out_path.exists():
        return set()
    completed = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed.add(
                (record["prompt_id"], record["label"], record["language"]))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--harmful-csv")
    parser.add_argument("--harmless-csv")
    parser.add_argument("--harmful-col", default="goal")
    parser.add_argument("--harmless-col", default="prompt")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        requests = load_smoke_test_dataset(
            Path(__file__).resolve().parent.parent / "data" /
            "smoke_test_prompts.json")
    elif args.harmful_csv and args.harmless_csv:
        requests = load_probe_dataset(args.harmful_csv, args.harmless_csv,
                                      args.harmful_col, args.harmless_col,
                                      args.max_per_class)
    else:
        raise SystemExit(
            "Pass either --smoke-test or both --harmful-csv and --harmless-csv."
        )

    if args.out is None:
        args.out = ("results/probe_generalization/translations_smoketest.jsonl"
                   if args.smoke_test else
                   "results/probe_generalization/translations.jsonl")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_resume and out_path.exists():
        out_path.unlink()
    completed = load_completed(out_path)
    if completed:
        print(f"Resuming: {len(completed)} (prompt, label, language) triples "
              f"already done, will be skipped.")

    index = []
    for req in requests:
        for language in TEST_LANGUAGES:
            if (req.id, req.label, language) in completed:
                continue
            index.append((req, language))

    if not index:
        print(
            "Nothing left to do — all requested translations are already in the output file."
        )
        return

    translator = get_translator()
    eng = LANGUAGE_CODES["english"]

    # DeepL translates per (source_lang, target_lang) pair in one batched
    # call each — group the index by target language.
    print(f"Translating {len(index)} prompts via DeepL ...")
    by_language: dict[str, list[int]] = {}
    for i, (_, language) in enumerate(index):
        by_language.setdefault(language, []).append(i)

    results: list = [None] * len(index)
    for language, positions in by_language.items():
        texts = [index[i][0].text for i in positions]
        translated = batch_translate(translator, texts, eng,
                                     LANGUAGE_CODES[language])
        for pos, t in zip(positions, translated):
            results[pos] = t

    with open(out_path, "a") as f:
        for (req, language), translated in zip(index, results):
            if isinstance(translated, Exception):
                print(f"[{req.id}] {language:10s} FAILED: {translated!r} — "
                      f"will retry on next run")
                continue

            has_unk_token = "<unk>" in translated
            record = {
                "prompt_id": req.id,
                "label": req.label,
                "language": language,
                "original": req.text,
                "translated": translated,
                "has_unk_token": has_unk_token,
                "needs_review": has_unk_token,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            flag = " NEEDS_REVIEW (unk token)" if has_unk_token else ""
            print(f"[{req.id}] {language:10s} translated{flag}")

    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
