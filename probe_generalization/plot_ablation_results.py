"""Plots refusal rate by ablated layer from an
ablation_summary__{model}__{metric}.csv (see summarize_ablation.py), saved
as a local PNG. --metric picks which summary file to read the same way
summarize_ablation.py picks which one to write (auto = judge if it exists,
else keyword) — always shown in both the title and the output filename, so
a saved PNG never leaves the metric ambiguous.

Usage:
    python3 probe_generalization/plot_ablation_results.py \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "english": "#2a78d6",
    "chinese": "#eb6834",
    "japanese": "#1baf7a",
    "spanish": "#eda100",
}


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results/probe_generalization")
    parser.add_argument("--metric", choices=["auto", "keyword", "judge"],
                        default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = model_slug(args.model)
    if args.metric == "auto":
        judge_path = Path(
            args.results_dir) / f"ablation_summary__{slug}__judge.csv"
        metric_name = "judge" if judge_path.exists() else "keyword"
    else:
        metric_name = args.metric
    csv_path = Path(
        args.results_dir) / f"ablation_summary__{slug}__{metric_name}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path} — run summarize_ablation.py "
                         f"--metric {metric_name} first.")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for lang, group in df.groupby("language"):
        group = group.sort_values("layer")
        color = COLORS.get(lang, "#888888")
        ax.plot(group["layer"], group["refusal_rate"],
               label=f"{lang.capitalize()} (ablated)", color=color,
               linewidth=2, marker="o", markersize=4)
        ax.axhline(group["baseline_refusal_rate"].iloc[0],
                  color=color, linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Refusal rate")
    ax.set_title(f"Refusal rate under single-layer ablation — {args.model}\n"
                f"metric={metric_name} (dashed = unablated baseline, "
                f"same color per language)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(
        args.results_dir) / f"ablation_summary__{slug}__{metric_name}.png"
    fig.savefig(out_path)
    print(f"Saved {out_path} (metric={metric_name})")


if __name__ == "__main__":
    main()
