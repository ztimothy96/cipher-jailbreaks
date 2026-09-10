"""Step C — causal ablation check: for each probe layer, 
ablate its diff-of-means direction and measure the drop in refusal rate across languages.

Usage:
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct \\
        --layers 3,8,14,20
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct \\
        --languages english,chinese,japanese,spanish
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import DEFAULT_MODEL, app
from probe_generalization.dataset import LabeledRequest
from probe_generalization.modal_app import AblationChatModel
from probe_generalization.prompt_rendering import (TEST_LANGUAGES,
                                                   build_prompt,
                                                   is_refusal_multilingual,
                                                   load_translations)
from refusal_gap.data import load_harmful_csv

BASELINE = "baseline"
ALL_LANGUAGES = ["english"] + TEST_LANGUAGES


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_completed(out_path: Path,
                   model_name: str) -> set[tuple[int, str, str]]:
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
            completed.add((record["prompt_id"], record["condition"],
                           record.get("language", "english")))
    return completed


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    probes_path: str = None,
    harmful_csv: str = "data/polyrefuse/harmful.csv",
    harmful_col: str = "goal",
    max_prompts: int = 20,
    max_new_tokens: int = 256,
    layers: str = None,
    languages: str = "english",
    translations_path: str = "results/probe_generalization/translations.jsonl",
    out: str = None,
    resume: bool = True,
):
    languages = languages.split(",")
    unknown = set(languages) - set(ALL_LANGUAGES)
    if unknown:
        raise SystemExit(
            f"Unknown language(s): {unknown}. Valid: {ALL_LANGUAGES}")
    translations = load_translations(
        Path(translations_path)) if set(languages) - {"english"} else {}
    slug = model_slug(model)
    if probes_path is None:
        probes_path = f"results/probe_generalization/probes/{slug}.npz"
    probes_path = Path(probes_path)
    if not probes_path.exists():
        raise SystemExit(
            f"Missing {probes_path} — run probe.py for this model first.")
    probe_data = np.load(probes_path)
    weights = probe_data[
        "weights"]  # (n_layers, hidden_dim); index 0 = embeddings
    n_probe_layers = weights.shape[0]

    if layers is None:
        # Layer 0 is the embedding output, so it's skipped.
        candidate_layers = list(range(1, n_probe_layers))
    else:
        candidate_layers = [int(x) for x in layers.split(",")]

    harmful_texts = load_harmful_csv(harmful_csv,
                                     column=harmful_col)[:max_prompts]
    requests = [
        LabeledRequest(id=i, text=t, label=1)
        for i, t in enumerate(harmful_texts)
    ]

    if out is None:
        out = f"results/probe_generalization/ablation__{slug}.jsonl"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
        out_path.unlink()
    completed = load_completed(out_path, model)
    if completed:
        print(f"Resuming: {len(completed)} (prompt, condition, language) "
              f"triples already done for model '{model}', will be skipped.")

    conditions = [BASELINE] + [str(l) for l in candidate_layers]

    index = []
    for lang in languages:
        for req in requests:
            try:
                system_prompt, encoded = build_prompt(lang, req, translations)
            except KeyError as e:
                print(f"[{lang}] skipping prompt {req.id}: {e}")
                continue
            for condition in conditions:
                if (req.id, condition, lang) in completed:
                    continue
                index.append((req.id, condition, lang, system_prompt, encoded))

    if not index:
        print("Nothing left to do — all requested (prompt, condition, "
              "language) triples are already in the output file.")
        return

    print(f"Dispatching {len(index)} generations to Modal (model={model}, "
          f"{len(candidate_layers)} layers + baseline, "
          f"{len(requests)} prompts, languages={languages}) ...")
    chat_model = AblationChatModel(model_name=model)

    system_prompts = [row[3] for row in index]
    user_turns = [row[4] for row in index]
    block_idxs = [
        None if condition == BASELINE else int(condition) - 1
        for _, condition, _, _, _ in index
    ]
    directions = [
        None if condition == BASELINE else weights[int(condition)].tolist()
        for _, condition, _, _, _ in index
    ]
    max_new_tokens_list = [max_new_tokens] * len(index)

    with open(out_path, "a") as f:
        results = chat_model.generate.map(
            system_prompts,
            user_turns,
            block_idxs,
            directions,
            max_new_tokens_list,
            order_outputs=True,
            return_exceptions=True,
        )

        for (prompt_id, condition, lang, system_prompt,
             encoded), completion in zip(index, results):
            if isinstance(completion, Exception):
                print(f"[{lang}] [{prompt_id}] {condition:>10s} FAILED "
                      f"after retries: {completion!r} — will retry on next "
                      f"--resume run")
                continue

            record = {
                "model": model,
                "prompt_id": prompt_id,
                "condition": condition,  # "baseline" or probe-layer index
                "language": lang,
                "prompt": encoded,
                "completion": completion,
                "is_refusal": is_refusal_multilingual(completion, lang),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{lang}] [{prompt_id}] {condition:>10s} "
                  f"refusal={record['is_refusal']!s:5}")

    print(f"\nWrote results to {out_path}")
