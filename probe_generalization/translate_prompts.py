"""Translate the probe dataset into Chinese/Japanese/Spanish via NLLB,
plus a back-translation round-tripcsimilarity check for flagging likely-bad 
translations for manual spot-check.

Usage:
    modal run probe_generalization/translate_prompts.py --smoke-test
    modal run probe_generalization/translate_prompts.py \\
        --harmful-csv path/to/advbench.csv --harmless-csv path/to/harmless.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import app
from probe_generalization.dataset import load_probe_dataset, load_smoke_test_dataset
from probe_generalization.formats import (FLORES_CODES, TEST_LANGUAGES,
                                          translation_round_trip_similarity)
from probe_generalization.modal_app import Translator


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


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    harmful_csv: str = None,
    harmless_csv: str = None,
    harmful_col: str = "goal",
    harmless_col: str = "prompt",
    max_per_class: int = None,
    out: str = None,
    resume: bool = True,
    similarity_flag_threshold: float = 0.3,
):
    if smoke_test:
        requests = load_smoke_test_dataset(
            Path(__file__).resolve().parent.parent / "data" /
            "smoke_test_prompts.json")
    elif harmful_csv and harmless_csv:
        requests = load_probe_dataset(harmful_csv, harmless_csv, harmful_col,
                                      harmless_col, max_per_class)
    else:
        raise SystemExit(
            "Pass either --smoke-test or both --harmful-csv and --harmless-csv."
        )

    if out is None:
        out = "results/probe_generalization/translations.jsonl"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
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

    print(f"Dispatching {len(index)} translations to Modal ...")
    translator = Translator()
    src_lang = FLORES_CODES["english"]

    # Forward pass: English -> target language.
    texts = [req.text for req, _ in index]
    src_langs = [src_lang] * len(index)
    tgt_langs = [FLORES_CODES[language] for _, language in index]
    forward_results = list(
        translator.translate.map(texts,
                                 src_langs,
                                 tgt_langs,
                                 order_outputs=True,
                                 return_exceptions=True))

    # Retry any failed forward translations.
    backward_entries = []  # (req, language, translated)
    for (req, language), translated in zip(index, forward_results):
        if isinstance(translated, Exception):
            print(f"[{req.id}] {language:10s} FAILED (forward translation): "
                  f"{translated!r} — will retry on next --resume run")
            continue
        backward_entries.append((req, language, translated))

    # Back-translation: target language -> English, for the quality check.
    if backward_entries:
        back_texts = [t for _, _, t in backward_entries]
        back_src_langs = [
            FLORES_CODES[lang] for _, lang, _ in backward_entries
        ]
        back_tgt_langs = [src_lang] * len(backward_entries)
        backward_results = list(
            translator.translate.map(back_texts,
                                     back_src_langs,
                                     back_tgt_langs,
                                     order_outputs=True,
                                     return_exceptions=True))
    else:
        backward_results = []

    with open(out_path, "a") as f:
        for (req, language, translated), back in zip(backward_entries,
                                                     backward_results):
            if isinstance(back, Exception):
                print(f"[{req.id}] {language:10s} FAILED (back-translation): "
                      f"{back!r} — will retry on next --resume run")
                continue

            similarity = translation_round_trip_similarity(req.text, back)
            needs_review = similarity < similarity_flag_threshold
            record = {
                "prompt_id": req.id,
                "label": req.label,
                "language": language,
                "original": req.text,
                "translated": translated,
                "back_translated": back,
                "round_trip_similarity": similarity,
                "needs_review": needs_review,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            flag = " NEEDS_REVIEW" if needs_review else ""
            print(
                f"[{req.id}] {language:10s} similarity={similarity:.2f}{flag}")

    print(f"\nWrote results to {out_path}")
