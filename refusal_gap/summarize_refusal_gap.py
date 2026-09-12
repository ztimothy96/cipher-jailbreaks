"""Summarizes measure_refusal_gap.py's output into a refusal-rate-per-cipher
bar chart — same idea as probe_generalization/summarize_ablation.py's
layer plot, but the x-axis is cipher instead of ablated layer, since this
measures a behavioral effect (surface encoding) rather than a causal
intervention (direction ablation).

Usage:
    python3 refusal_gap/summarize_refusal_gap.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results/refusal_gap")
    parser.add_argument(
        "--path",
        default=None,
        help="Defaults to results-dir/refusal_gap__<model_slug>.jsonl")
    args = parser.parse_args()

    slug = model_slug(args.model)
    path = Path(args.path
               or Path(args.results_dir) / f"refusal_gap__{slug}.jsonl")
    if not path.exists():
        raise SystemExit(f"Missing {path} — run measure_refusal_gap.py first.")

    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]

    rates = df.groupby("cipher")["is_refusal"].mean()
    n = df.groupby("cipher")["is_refusal"].size()
    decode_ok = df.groupby("cipher")["decode_looks_valid"].mean()
    out = pd.DataFrame({
        "refusal_rate": rates,
        "decode_ok_rate": decode_ok,
        "n": n,
    }).sort_values("refusal_rate", ascending=False)

    baseline_rate = rates.get("plaintext")

    print(out.to_string(
        formatters={
            "refusal_rate": "{:.2%}".format,
            "decode_ok_rate": "{:.2%}".format,
        }))
    if baseline_rate is None:
        print("\nNote: no 'plaintext' condition in this file — no baseline "
              "reference line to plot. Include plaintext in --ciphers on a "
              "future run to get one.")

    out_path = Path(args.results_dir) / f"refusal_gap_summary__{slug}.csv"
    out.to_csv(out_path)
    print(f"\nSaved {out_path}")

    ciphers = out.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(ciphers, out["refusal_rate"], color="#2a78d6")
    if baseline_rate is not None:
        ax.axhline(baseline_rate,
                   color="#888888",
                   linewidth=1,
                   linestyle="--",
                   alpha=0.8,
                   label="plaintext (unciphered)")
        ax.legend(loc="lower right")

    ax.set_xlabel("Cipher")
    ax.set_ylabel("Refusal rate")
    ax.set_title(f"Refusal rate by cipher — {args.model}\n"
                 f"metric=keyword scan (dashed = plaintext baseline, "
                 f"if present)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    png_path = Path(args.results_dir) / f"refusal_gap_summary__{slug}.png"
    fig.savefig(png_path)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
