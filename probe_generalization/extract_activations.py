"""Step A2/B2 — activation extraction: one forward pass per (prompt, format)
per model, saving the last-token residual stream at every layer (see
ActivationExtractor.extract in modal_app.py).

Usage:
    modal run probe_generalization/extract_activations.py --smoke-test \\
        --model Qwen/Qwen2.5-7B-Instruct --format-group languages
    modal run probe_generalization/extract_activations.py --smoke-test \\
        --model Qwen/Qwen2.5-7B-Instruct --format-group ciphers
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import load_probe_dataset, load_smoke_test_dataset
from common.modal_infra import DEFAULT_MODEL, app, model_slug
from probe_generalization.shared.formats import TEST_LANGUAGES
from probe_generalization.shared.modal_app import ActivationExtractor
from probe_generalization.shared.prompt_rendering import (CIPHER_NAMES, build_prompt,
                                                   load_translations)

FORMATS_BY_GROUP = {
    "languages": ["english"] + TEST_LANGUAGES,
    "ciphers": CIPHER_NAMES,
}


def raw_path(raw_dir: Path, label: int, prompt_id: int) -> Path:
    return raw_dir / f"{label}_{prompt_id}.npy"


def consolidate(raw_dir: Path, out_path: Path) -> int:
    """Packs every per-prompt .npy file currently in raw_dir into out_path
    (activations, prompt_ids, labels), overwriting it. Safe to call with
    zero new files — just re-packs whatever's already there. Returns the
    number of prompts packed."""
    ids, labels, vectors = [], [], []
    for path in sorted(raw_dir.glob("*.npy")):
        label, prompt_id = (int(x) for x in path.stem.split("_", 1))
        vectors.append(np.load(path))
        ids.append(prompt_id)
        labels.append(label)

    if not vectors:
        return 0

    activations = np.stack(vectors)  # (n_prompts, n_layers+1, hidden_dim)
    np.savez_compressed(out_path,
                        activations=activations,
                        prompt_ids=np.array(ids),
                        labels=np.array(labels))
    return len(vectors)


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    harmful_csv: str = None,
    harmless_csv: str = None,
    harmful_col: str = "goal",
    harmless_col: str = "prompt",
    max_per_class: int = None,
    model: str = DEFAULT_MODEL,
    format_group: str = "languages",
    out_dir: str = None,
    resume: bool = True,
    translations_path: str = None,
):
    if format_group not in FORMATS_BY_GROUP:
        raise SystemExit(
            f"--format-group must be one of {list(FORMATS_BY_GROUP)}.")

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

    formats = FORMATS_BY_GROUP[format_group]
    if translations_path is None:
        translations_path = (
            "results/probe_generalization/translations_smoketest.jsonl" if
            smoke_test else "results/probe_generalization/translations.jsonl")
    translations = load_translations(
        Path(translations_path)) if format_group == "languages" else {}

    if out_dir is None:
        suffix = "_smoketest" if smoke_test else ""
        out_dir = f"results/probe_generalization/activations{suffix}__{format_group}"
    out_dir_path = Path(out_dir) / model_slug(model)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    extractor = ActivationExtractor(model_name=model)

    for fmt in formats:
        out_path = out_dir_path / f"{fmt}.npz"
        raw_dir = out_dir_path / fmt / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        prompts = []
        for req in requests:
            if resume and raw_path(raw_dir, req.label, req.id).exists():
                continue
            try:
                system_prompt, encoded = build_prompt(fmt, req, translations)
            except KeyError as e:
                print(f"[{fmt}] skipping prompt {req.id}: {e}")
                continue
            prompts.append((req, system_prompt, encoded))

        if not prompts:
            print(f"[{fmt}] nothing left to extract.")
        else:
            print(f"[{fmt}] dispatching {len(prompts)} forward passes to "
                  f"Modal (model={model}) ...")
            system_prompts = [p[1] for p in prompts]
            user_turns = [p[2] for p in prompts]
            # Not list()-wrapped: iterate the .map() generator directly so
            # each result is saved to disk as soon as it arrives, rather
            # than buffering the whole batch in memory until it's done.
            results = extractor.extract.map(system_prompts,
                                            user_turns,
                                            order_outputs=True,
                                            return_exceptions=True)

            for (req, _, _), result in zip(prompts, results):
                if isinstance(result, Exception):
                    print(f"[{fmt}] prompt {req.id} FAILED: {result!r} — "
                          f"rerun to retry")
                    continue
                np.save(raw_path(raw_dir, req.label, req.id), result)

        n_packed = consolidate(raw_dir, out_path)
        print(f"[{fmt}] {n_packed} prompts packed into {out_path}")

    print("\nDone.")
