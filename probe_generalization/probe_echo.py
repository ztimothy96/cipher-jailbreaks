"""Trains a diff-of-means probe per layer to predict ECHO vs. non-ECHO from
the prompt's activations alone — tests whether echo behavior (the model
reproducing the cipher-formatted request instead of responding to it) is
predictable from the input representation, or only emerges during
generation. Reuses the same activations extract_activations.py already
wrote for the refusal probe (probe.py) — ECHO is a property of a
deterministic (do_sample=False) generation from a fixed prompt, so no new
Modal run is needed, only the labels from groq_judge_ablation.py's output.

Small-n (order ~100 prompts) relative to hidden_dim, so a plain train/test
split would be noisy — this cross-validates instead, aggregating
out-of-fold scores across folds before computing one AUROC per layer,
rather than averaging per-fold AUROCs (which can be dominated by the very
few ECHO examples landing in any given fold).

Usage:
    python3 probe_generalization/probe_echo.py --model Qwen/Qwen2.5-7B-Instruct
    python3 probe_generalization/probe_echo.py --model Qwen/Qwen2.5-7B-Instruct \\
        --cipher letter_spaced --condition baseline --max-prompts 100
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug
from probe_generalization.shared.prompt_rendering import FORMAT_GROUP

SEED = 0
N_FOLDS = 5


def load_echo_labels(judged_path: Path, cipher: str, condition: str,
                     max_prompts: int) -> dict[int, int]:
    labels = {}
    with open(judged_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["cipher"] != cipher or r["condition"] != condition:
                continue
            if max_prompts is not None and r["prompt_id"] >= max_prompts:
                continue
            labels[r["prompt_id"]] = int(r["judge_label"] == "ECHO")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cipher", default="letter_spaced")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--activations-root",
                        default="results/probe_generalization")
    parser.add_argument("--results-dir", default="results/probe_generalization")
    args = parser.parse_args()

    slug = model_slug(args.model)
    track = FORMAT_GROUP[args.cipher]

    judged_path = Path(
        args.results_dir) / f"ablation_judged_{track}__{slug}.jsonl"
    if not judged_path.exists():
        raise SystemExit(f"Missing {judged_path} — run ablate.py and "
                         f"groq_judge_ablation.py first.")
    echo_labels = load_echo_labels(judged_path, args.cipher, args.condition,
                                   args.max_prompts)
    if not echo_labels:
        raise SystemExit(f"No judged records for cipher={args.cipher} "
                         f"condition={args.condition} in {judged_path}.")
    n_echo = sum(echo_labels.values())
    print(f"Loaded {len(echo_labels)} judged prompts "
         f"({n_echo} ECHO, {len(echo_labels) - n_echo} non-ECHO) from "
         f"{judged_path}\n")

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
        i for i, pid in enumerate(prompt_ids) if int(pid) in echo_labels
    ]
    missing = set(echo_labels) - set(int(prompt_ids[i]) for i in order)
    if missing:
        print(f"Warning: {len(missing)} judged prompt_id(s) have no "
             f"matching activation and will be dropped: {sorted(missing)}")
    order.sort(key=lambda i: int(prompt_ids[i]))

    X = activations[order]  # (n, n_layers, hidden_dim)
    y = np.array([echo_labels[int(prompt_ids[i])] for i in order])
    n_samples, n_layers, hidden_dim = X.shape
    print(f"Aligned {n_samples} (activation, label) pairs across "
         f"{n_layers} layers.\n")

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
        auroc = roc_auc_score(y, oof_scores)
        rows.append({"layer": layer, "auroc": auroc, "n": n_samples})

    out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print(out.to_string(
        index=False, formatters={"auroc": "{:.3f}".format}))

    out_path = Path(
        args.results_dir
    ) / f"echo_probe__{slug}__{args.cipher}__{args.condition}.csv"
    out.sort_values("layer").to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    plot_df = out.sort_values("layer")
    ax.plot(plot_df["layer"], plot_df["auroc"], color="#1baf7a", linewidth=2,
           marker="o", markersize=4)
    ax.axhline(0.5, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (5-fold CV, out-of-fold)")
    ax.set_title(f"ECHO-vs-not probe by layer — {args.model}\n"
                f"cipher={args.cipher} condition={args.condition} "
                f"(n={n_samples}, {n_echo} ECHO)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = Path(
        args.results_dir
    ) / f"echo_probe__{slug}__{args.cipher}__{args.condition}.png"
    fig.savefig(png_path)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
