"""Plots refusal rate by ablated layer from an ablation_summary__*.csv (see
summarize_ablation.py), saved as a local PNG.

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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = model_slug(args.model)
    csv_path = Path(args.results_dir) / f"ablation_summary__{slug}.csv"
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
                f"(dashed = unablated baseline, same color per language)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(
        args.results_dir) / f"ablation_summary__{slug}.png"
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
