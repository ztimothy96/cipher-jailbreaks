"""Step A3/A4 + B3 — probe training and cross-format evaluation.

Trains one diff-of-means direction per (model, layer) on a
held-out 80/20 stratified split of the English activations, evaluates AUROC
on the held-out English test split and zero-shot on other formats'
activation sets (languages and/or ciphers) using the same direction.

Retraining is deterministic (fixed SEED, pure local numpy over cached
activations, no GPU/Modal involved) — a fresh run reproduces the exact same
probe weights already saved by a prior run, so this doubles as Step B3
("zero-shot AUROC of the *same* probes ... no retraining", per
docs/probe-generalization-plan.md): evaluating a new format here doesn't
need to load a previously-saved probe from disk to satisfy that.

Usage:
    python3 probe_generalization/probe.py --model Qwen/Qwen2.5-7B-Instruct
    python3 probe_generalization/probe.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats letter_spaced
    python3 probe_generalization/probe.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats chinese,japanese,spanish,letter_spaced
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
from probe_generalization.shared.prompt_rendering import ALL_FORMATS, FORMAT_GROUP

SEED = 0
TEST_SIZE = 0.2
DEFAULT_FORMATS = ["chinese", "japanese", "spanish"]


def load_npz(activations_root: Path, fmt: str, model: str) -> dict:
    path = activations_root / f"activations__{FORMAT_GROUP[fmt]}" / model_slug(
        model) / f"{fmt}.npz"
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
        "--activations-root",
        default="results/probe_generalization",
        help="Parent of the activations__languages/activations__ciphers "
        "directories written by extract_activations.py.")
    parser.add_argument("--out-dir", default="results/probe_generalization")
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated subset of languages/ciphers to zero-shot "
        "evaluate against the English-trained probe, e.g. while others are "
        "still being extracted.")
    args = parser.parse_args()
    formats = args.formats.split(",")
    unknown = set(formats) - (set(ALL_FORMATS) - {"english"})
    if unknown:
        raise SystemExit(f"Unknown format(s): {unknown}. Valid: "
                         f"{sorted(set(ALL_FORMATS) - {'english'})}")

    activations_root = Path(args.activations_root)
    english = load_npz(activations_root, "english", args.model)
    others = {
        fmt: load_npz(activations_root, fmt, args.model)
        for fmt in formats
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

        for fmt, data in others.items():
            X_fmt = data["activations"][:, layer, :]
            y_fmt = data["labels"]
            auroc_fmt = roc_auc_score(y_fmt, score(X_fmt))
            rows.append({
                "model": args.model,
                "layer": layer,
                "format": fmt,
                "auroc": auroc_fmt,
                "n": len(y_fmt),
            })

        print(f"layer {layer:3d}: english={auroc_en:.3f} " +
              " ".join(f"{fmt}={r['auroc']:.3f}"
                       for fmt, r in zip(others, rows[-len(others):])))

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
    new_df = pd.DataFrame(rows)
    # Merge rather than overwrite: this run only recomputed rows for
    # `args.model` and formats in {"english"} | set(formats) — replace just
    # those in the existing file (if any) and leave every other model/format
    # combination already on disk untouched, so re-running with a narrower
    # --formats doesn't silently drop previously-saved results.
    if results_path.exists():
        old_df = pd.read_csv(results_path)
        recomputed = (old_df["model"] == args.model) & (
            old_df["format"].isin({"english", *formats}))
        new_df = pd.concat([old_df[~recomputed], new_df], ignore_index=True)
    new_df = new_df.sort_values(["model", "format", "layer"])
    new_df.to_csv(results_path, index=False)
    print(f"Saved AUROC results to {results_path}")


if __name__ == "__main__":
    main()
