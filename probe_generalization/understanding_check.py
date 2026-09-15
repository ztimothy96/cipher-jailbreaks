"""Step A1/B1 — decode/understanding check: confirm the model actually
engages with a harmful/harmless request in a given non-English-plaintext
format (language or cipher) before trusting any probe result on that format.
A format the model can't parse isn't a probe-generalization result, it's a
capability gap. Mirrors refusal_gap's decode_looks_valid check. Resumable,
same pattern as refusal_gap/measure_refusal_gap.py.

Prompt-rendering logic (build_prompt, load_translations, format lists) lives
in prompt_rendering.py, not here — extract_activations.py needs the same
rendering and can't import this file directly without colliding with its
own @app.local_entrypoint() (see prompt_rendering.py docstring).

Usage:
    modal run probe_generalization/understanding_check.py --smoke-test \\
        --model Qwen/Qwen2.5-7B-Instruct --formats english chinese
    modal run probe_generalization/understanding_check.py --smoke-test \\
        --model Qwen/Qwen2.5-7B-Instruct --formats rot13 base64 leetspeak
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.chat_model import ChatModel
from common.dataset import load_probe_dataset, load_smoke_test_dataset
from common.modal_infra import DEFAULT_MODEL, app
from probe_generalization.shared.prompt_rendering import (ALL_FORMATS, build_prompt,
                                                        is_refusal_multilingual,
                                                        load_translations,
                                                        looks_like_noise)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_completed(out_path: Path,
                   model_name: str) -> set[tuple[int, int, str]]:
    if not out_path.exists():
        return set()
    completed = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("model") != model_name:
                continue
            completed.add((record["prompt_id"], record["label"], record["format"]))
    return completed


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    harmful_csv: str = None,
    harmless_csv: str = None,
    harmful_col: str = "goal",
    harmless_col: str = "prompt",
    max_per_class: int = None,
    model: str = DEFAULT_MODEL,
    formats: str = None,  # comma-separated subset of ALL_FORMATS; default = all
    max_new_tokens: int = 64,  # a coherence check needs a sentence, not a full answer
    translations_path: str = None,
    out: str = None,
    resume: bool = True,
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

    format_list = formats.split(",") if formats else ALL_FORMATS
    unknown = set(format_list) - set(ALL_FORMATS)
    if unknown:
        raise SystemExit(f"Unknown format(s): {unknown}. Valid: {ALL_FORMATS}")

    if translations_path is None:
        translations_path = (
            "results/probe_generalization/translations_smoketest.jsonl"
            if smoke_test else
            "results/probe_generalization/translations.jsonl")
    translations = load_translations(Path(translations_path))

    if out is None:
        suffix = "_smoketest" if smoke_test else ""
        out = f"results/probe_generalization/understanding_check{suffix}__{model_slug(model)}.jsonl"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
        out_path.unlink()
    completed = load_completed(out_path, model)
    if completed:
        print(f"Resuming: {len(completed)} (prompt, label, format) triples "
              f"already done for model '{model}', will be skipped.")

    index = []
    for req in requests:
        for fmt in format_list:
            if (req.id, req.label, fmt) in completed:
                continue
            try:
                system_prompt, encoded = build_prompt(fmt, req, translations)
            except KeyError as e:
                print(f"[{req.id}] {fmt:10s} SKIPPED: {e}")
                continue
            index.append((req, fmt, system_prompt, encoded))

    if not index:
        print(
            "Nothing left to do — all requested (prompt, format) pairs are already in the output file."
        )
        return

    print(f"Dispatching {len(index)} generations to Modal (model={model}) ...")
    chat_model = ChatModel(model_name=model)

    system_prompts = [row[2] for row in index]
    user_turns = [row[3] for row in index]
    max_new_tokens_list = [max_new_tokens] * len(index)

    with open(out_path, "a") as f:
        results = chat_model.generate.map(system_prompts,
                                          user_turns,
                                          max_new_tokens_list,
                                          order_outputs=True,
                                          return_exceptions=True)

        for (req, fmt, system_prompt, encoded), completion in zip(index, results):
            if isinstance(completion, Exception):
                print(f"[{req.id}] {fmt:10s} FAILED after retries: "
                      f"{completion!r} — will retry on next --resume run")
                continue

            record = {
                "model": model,
                "prompt_id": req.id,
                "label": req.label,
                "format": fmt,
                "system_prompt": system_prompt,
                "encoded_request": encoded,
                "completion": completion,
                "understanding_ok": not looks_like_noise(completion),
                "is_refusal": is_refusal_multilingual(completion, fmt),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{req.id}] {fmt:10s} understanding_ok="
                  f"{record['understanding_ok']!s:5} refusal={record['is_refusal']!s:5}")

    print(f"\nWrote results to {out_path}")
