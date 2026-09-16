"""Step C — causal ablation check: for each probe layer,
ablate its diff-of-means direction and measure the drop in refusal rate
across languages and/or ciphers.

Usage:
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct \\
        --layers 3,8,14,20
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats english,chinese,japanese,spanish
    modal run probe_generalization/ablate.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats letter_spaced
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.ciphers import ALL_CIPHERS
from common.data import load_harmful_csv
from common.dataset import LabeledRequest
from common.jsonl_output import sort_output_file
from common.modal_infra import DEFAULT_MODEL, app, gpu_for, model_slug
from probe_generalization.shared.ablation_records import (
    BASELINE, FMT_FIELD, CipherAblationRecord, LanguageAblationRecord,
    sort_key)
from probe_generalization.shared.modal_app import AblationChatModel
from probe_generalization.shared.prompt_rendering import (ALL_FORMATS,
                                                          FORMAT_GROUP,
                                                          TEST_LANGUAGES,
                                                          build_prompt,
                                                          load_translations)


def decode_for_record(fmt: str, text: str) -> str:
    """Decodes a cipher completion back to plaintext for storage."""
    try:
        return ALL_CIPHERS[fmt].decode(text)
    except Exception:
        return text


def load_completed(out_path: Path, model_name: str,
                   fmt_field: str) -> set[tuple[int, str, str]]:
    """Loads completed records from an output file."""
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
            completed.add(
                (record["prompt_id"], record["condition"], record[fmt_field]))
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
    formats: str = "english",
    translations_path: str = "results/probe_generalization/translations.jsonl",
    out_dir: str = "results/probe_generalization",
    resume: bool = True,
):
    formats = formats.split(",")
    unknown = set(formats) - set(ALL_FORMATS)
    if unknown:
        raise SystemExit(f"Unknown format(s): {unknown}. Valid: {ALL_FORMATS}")
    translations = load_translations(
        Path(translations_path)) if set(formats) & set(TEST_LANGUAGES) else {}
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

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_paths = {
        track: out_dir_path / f"ablation_{track}__{slug}.jsonl"
        for track in ("languages", "ciphers")
    }

    if not resume:
        for path in out_paths.values():
            if path.exists():
                path.unlink()
    completed = {
        track: load_completed(path, model, FMT_FIELD[track])
        for track, path in out_paths.items()
    }
    n_completed = sum(len(c) for c in completed.values())
    if n_completed:
        print(f"Resuming: {n_completed} (prompt, condition, format) "
              f"triples already done for model '{model}', will be skipped.")

    conditions = [BASELINE] + [str(l) for l in candidate_layers]

    index = []
    for fmt in formats:
        track = FORMAT_GROUP[fmt]
        for req in requests:
            try:
                system_prompt, encoded = build_prompt(fmt, req, translations)
            except KeyError as e:
                print(f"[{fmt}] skipping prompt {req.id}: {e}")
                continue
            for condition in conditions:
                if (req.id, condition, fmt) in completed[track]:
                    continue
                index.append(
                    (req, condition, fmt, track, system_prompt, encoded))

    if not index:
        print("Nothing left to do — all requested (prompt, condition, "
              "format) triples are already in the output file.")
        for track, path in out_paths.items():
            sort_output_file(path, key=sort_key(FMT_FIELD[track]))
        return

    print(f"Dispatching {len(index)} generations to Modal (model={model}, "
          f"{len(candidate_layers)} layers + baseline, "
          f"{len(requests)} prompts, formats={formats}) ...")
    chat_model = AblationChatModel.with_options(gpu=gpu_for(model))(
        model_name=model)

    system_prompts = [row[4] for row in index]
    user_turns = [row[5] for row in index]
    block_idxs = [
        None if condition == BASELINE else int(condition) - 1
        for _, condition, _, _, _, _ in index
    ]
    directions = [
        None if condition == BASELINE else weights[int(condition)].tolist()
        for _, condition, _, _, _, _ in index
    ]
    max_new_tokens_list = [max_new_tokens] * len(index)

    open_files = {track: open(path, "a") for track, path in out_paths.items()}
    try:
        results = chat_model.generate.map(
            system_prompts,
            user_turns,
            block_idxs,
            directions,
            max_new_tokens_list,
            order_outputs=True,
            return_exceptions=True,
        )

        for (req, condition, fmt, track, system_prompt,
             encoded), completion in zip(index, results):
            if isinstance(completion, Exception):
                print(f"[{fmt}] [{req.id}] {condition:>10s} FAILED "
                      f"after retries: {completion!r} — will retry on next "
                      f"--resume run")
                continue

            if track == "ciphers":
                record = CipherAblationRecord(
                    model=model,
                    prompt_id=req.id,
                    condition=condition,
                    cipher=fmt,
                    encoded_prompt=encoded,
                    decoded_prompt=req.text,
                    raw_completion=completion,
                    decoded_completion=decode_for_record(fmt, completion),
                )
            else:
                record = LanguageAblationRecord(
                    model=model,
                    prompt_id=req.id,
                    condition=condition,
                    language=fmt,
                    prompt=encoded,
                    completion=completion,
                )

            f = open_files[track]
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{fmt}] [{req.id}] {condition:>10s} generated")
    finally:
        for f in open_files.values():
            f.close()

    for track, path in out_paths.items():
        sort_output_file(path, key=sort_key(FMT_FIELD[track]))
        print(f"Wrote {track} results to {path}")
