"""Plots AUROC by layer per format from a probe_results__*.csv, saved as a
local PNG.

Usage:
    python3 probe_generalization/plot_probe_results.py \\
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
    csv_path = Path(args.results_dir) / f"probe_results__{slug}.csv"
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for fmt, group in df.groupby("format"):
        group = group.sort_values("layer")
        ax.plot(group["layer"], group["auroc"], label=fmt.capitalize(),
               color=COLORS.get(fmt, "#888888"), linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Probe transfer by layer — {args.model}")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(
        args.results_dir) / f"probe_results__{slug}.png"
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
