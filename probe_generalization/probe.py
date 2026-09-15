"""Step A3/A4 — probe training and cross-language evaluation.

Trains one diff-of-means direction per (model, layer) on a
held-out 80/20 stratified split of the English activations, evaluates AUROC
on the held-out English test split and zero-shot on the other languages'
activation sets using the same direction.

Persists each layer's probe weights (not just AUROC numbers) — Phase B
(probe.py's cipher counterpart, not yet written) reuses these exact probes
rather than retraining, per docs/probe-generalization-plan.md Step B3.

Usage:
    python3 probe_generalization/probe.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug

SEED = 0
TEST_SIZE = 0.2
ALL_LANGUAGES = ["chinese", "japanese", "spanish"]


def load_npz(activations_dir: Path, model: str, fmt: str) -> dict:
    path = activations_dir / model_slug(model) / f"{fmt}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path} — run extract_activations.py first.")
    data = np.load(path)
    return {
        "activations": data["activations"],
        "labels": data["labels"],
        "prompt_ids": data["prompt_ids"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--activations-dir",
        default="results/probe_generalization/activations__languages")
    parser.add_argument("--out-dir", default="results/probe_generalization")
    parser.add_argument(
        "--languages",
        default=",".join(ALL_LANGUAGES),
        help="Comma-separated subset to evaluate, e.g. while others are "
        "still being extracted.")
    args = parser.parse_args()
    languages = args.languages.split(",")
    unknown = set(languages) - set(ALL_LANGUAGES)
    if unknown:
        raise SystemExit(
            f"Unknown language(s): {unknown}. Valid: {ALL_LANGUAGES}")

    activations_dir = Path(args.activations_dir)
    english = load_npz(activations_dir, args.model, "english")
    others = {
        lang: load_npz(activations_dir, args.model, lang)
        for lang in languages
    }

    X_en, y_en = english["activations"], english["labels"]
    n_prompts, n_layers, hidden_dim = X_en.shape

    train_idx, test_idx = train_test_split(np.arange(n_prompts),
                                           test_size=TEST_SIZE,
                                           stratify=y_en,
                                           random_state=SEED)
    y_train, y_test = y_en[train_idx], y_en[test_idx]

    weights = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    biases = np.zeros(n_layers, dtype=np.float32)
    rows = []

    for layer in range(n_layers):
        X_train = X_en[train_idx, layer, :]
        X_test = X_en[test_idx, layer, :]

        mu_pos = X_train[y_train == 1].mean(axis=0)
        mu_neg = X_train[y_train == 0].mean(axis=0)
        direction = mu_pos - mu_neg
        norm = np.linalg.norm(direction)
        # Layer 0 is identical across every prompt. Leave direction as the zero vector.
        if norm > 0:
            direction /= norm
        bias = -direction @ ((mu_pos + mu_neg) / 2)

        weights[layer] = direction
        biases[layer] = bias

        def score(X, direction=direction, bias=bias):
            return X @ direction + bias

        auroc_en = roc_auc_score(y_test, score(X_test))
        rows.append({
            "model": args.model,
            "layer": layer,
            "format": "english",
            "auroc": auroc_en,
            "n": len(test_idx),
        })

        for lang, data in others.items():
            X_lang = data["activations"][:, layer, :]
            y_lang = data["labels"]
            auroc_lang = roc_auc_score(y_lang, score(X_lang))
            rows.append({
                "model": args.model,
                "layer": layer,
                "format": lang,
                "auroc": auroc_lang,
                "n": len(y_lang),
            })

        print(f"layer {layer:3d}: english={auroc_en:.3f} " +
              " ".join(f"{lang}={r['auroc']:.3f}"
                       for lang, r in zip(others, rows[-len(others):])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes_path = out_dir / "probes" / f"{model_slug(args.model)}.npz"
    probes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probes_path,
                        weights=weights,
                        biases=biases,
                        train_idx=train_idx,
                        test_idx=test_idx)
    print(f"\nSaved probe weights to {probes_path}")

    results_path = out_dir / f"probe_results__{model_slug(args.model)}.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved AUROC results to {results_path}")


if __name__ == "__main__":
    main()
