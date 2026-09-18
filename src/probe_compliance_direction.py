"""Trains a diff-of-means probe directly on a cipher's own activations to
predict REFUSE vs COMPLY — unlike probe_compliance.py, which zero-shots the
*English*-trained harmfulness direction onto cipher activations, this fits
a fresh direction from the cipher's own REFUSE/COMPLY labels. Answers two
questions raised by probe_compliance.py's result (the English direction
predicts letter_spaced refusal well at late layers, but every score stays
on the "looks benign" side of the English decision boundary):

  1. Does a direction found *in this format* separate REFUSE from COMPLY
     well? 5-fold CV, out-of-fold AUROC — same small-n approach as
     probe_echo.py (this and that script hit the same ~69-example, few-
     dozen-positives regime).
  2. If so, is it geometrically the same direction as the English
     harmfulness probe (probe.py), or something different? Measured via
     cosine similarity between the two directions, each fit on its full
     available data (not per-fold — a stable single direction per layer is
     what we want to compare geometrically; the CV split above is only for
     getting an unbiased AUROC, a different concern).

Usage:
    python3 src/probe_compliance_direction.py --model Qwen/Qwen2.5-7B-Instruct
    python3 src/probe_compliance_direction.py --model Qwen/Qwen2.5-7B-Instruct \\
        --cipher letter_spaced --condition baseline
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug
from src.probe_compliance import VALID_JUDGE_LABELS, load_judge_labels
from src.shared.prompt_rendering import FORMAT_GROUP

SEED = 0
N_FOLDS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cipher", default="letter_spaced")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--activations-root", default="results")
    parser.add_argument("--probes-root", default="results/probes")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    slug = model_slug(args.model)
    track = FORMAT_GROUP[args.cipher]
    results_dir = Path(args.results_dir)

    pid_to_judge_label = load_judge_labels(results_dir, args.model, track,
                                           args.cipher, {args.condition},
                                           VALID_JUDGE_LABELS)
    if not pid_to_judge_label:
        raise SystemExit(f"No COMPLY/REFUSE-judged rows found for "
                         f"cipher={args.cipher} condition={args.condition}.")
    refuse_labels = {
        pid: int(label == "REFUSE")
        for pid, label in pid_to_judge_label.items()
    }
    n_refuse = sum(refuse_labels.values())
    print(f"Loaded {len(refuse_labels)} judged prompts "
         f"({n_refuse} REFUSE, {len(refuse_labels) - n_refuse} COMPLY) from "
         f"{results_dir / 'raw'}\n")

    activations_path = (Path(args.activations_root) /
                        f"activations__{track}" / slug / f"{args.cipher}.npz")
    if not activations_path.exists():
        raise SystemExit(f"Missing {activations_path} — run "
                         f"extract_activations.py for this cipher first.")
    data = np.load(activations_path)
    harmful_mask = data["labels"] == 1
    prompt_ids = data["prompt_ids"][harmful_mask]
    activations = data["activations"][harmful_mask]  # (n, n_layers, hidden)

    order = [
        i for i, pid in enumerate(prompt_ids) if int(pid) in refuse_labels
    ]
    missing = set(refuse_labels) - set(int(prompt_ids[i]) for i in order)
    if missing:
        print(f"Warning: {len(missing)} judged prompt_id(s) have no "
             f"matching activation and will be dropped: {sorted(missing)}")
    order.sort(key=lambda i: int(prompt_ids[i]))

    X = activations[order]  # (n, n_layers, hidden_dim)
    y = np.array([refuse_labels[int(prompt_ids[i])] for i in order])
    n_samples, n_layers, hidden_dim = X.shape
    print(f"Aligned {n_samples} (activation, label) pairs across "
         f"{n_layers} layers.\n")

    probes_path = Path(args.probes_root) / f"{slug}.npz"
    if not probes_path.exists():
        raise SystemExit(f"Missing {probes_path} — run probe.py first.")
    english_weights = np.load(probes_path)["weights"]  # (n_layers, hidden)
    if english_weights.shape[0] != n_layers:
        raise SystemExit(f"Layer count mismatch: English probe has "
                         f"{english_weights.shape[0]} layers, {args.cipher} "
                         f"activations have {n_layers}.")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for layer in range(n_layers):
        X_layer = X[:, layer, :]

        oof_scores = np.zeros(n_samples)
        for train_idx, test_idx in skf.split(X_layer, y):
            y_train = y[train_idx]
            mu_pos = X_layer[train_idx][y_train == 1].mean(axis=0)
            mu_neg = X_layer[train_idx][y_train == 0].mean(axis=0)
            direction = mu_pos - mu_neg
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            oof_scores[test_idx] = X_layer[test_idx] @ direction
        cv_auroc = roc_auc_score(y, oof_scores)

        # Full-data direction (all n_samples), for the geometry comparison —
        # a single stable estimate per layer, not one per CV fold.
        mu_pos_full = X_layer[y == 1].mean(axis=0)
        mu_neg_full = X_layer[y == 0].mean(axis=0)
        full_direction = mu_pos_full - mu_neg_full
        full_norm = np.linalg.norm(full_direction)
        if full_norm > 0:
            full_direction = full_direction / full_norm

        english_direction = english_weights[layer]
        cosine_sim = float(full_direction @ english_direction)

        rows.append({
            "model": args.model,
            "cipher": args.cipher,
            "condition": args.condition,
            "layer": layer,
            "cipher_refusal_auroc": cv_auroc,
            "cosine_sim_with_english_direction": cosine_sim,
            "n": n_samples,
        })

    out = pd.DataFrame(rows)
    print(out.sort_values("cipher_refusal_auroc", ascending=False).to_string(
        index=False,
        formatters={
            "cipher_refusal_auroc": "{:.3f}".format,
            "cosine_sim_with_english_direction": "{:+.3f}".format,
        }))

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / (
        f"probe_compliance_direction__{slug}__{args.cipher}__"
        f"{args.condition}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    ax.plot(out["layer"], out["cipher_refusal_auroc"], color="#1baf7a",
           linewidth=2, marker="o", markersize=4)
    ax.axhline(0.5, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (5-fold CV, out-of-fold)")
    ax.set_title(f"{args.cipher}-trained REFUSE-vs-COMPLY probe")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(out["layer"], out["cosine_sim_with_english_direction"],
           color="#4a3aa7", linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("...vs. English-trained harmfulness direction")
    ax.set_ylim(-1.02, 1.02)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{args.model} — cipher={args.cipher} "
                f"condition={args.condition} (n={n_samples}, "
                f"{n_refuse} REFUSE)")
    fig.tight_layout()

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / (
        f"probe_compliance_direction__{slug}__{args.cipher}__"
        f"{args.condition}.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
