"""Summarizes ablate.py's output into a refusal-rate-per-layer table, ranked
by drop from baseline — the layers most causally load-bearing for refusal
sort to the top.

Usage:
    python3 probe_generalization/summarize_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results/probe_generalization")
    args = parser.parse_args()

    slug = model_slug(args.model)
    path = Path(args.results_dir) / f"ablation__{slug}.jsonl"
    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]

    rates = df.groupby("condition")["is_refusal"].mean()
    n = df.groupby("condition")["is_refusal"].size()
    baseline_rate = rates.get("baseline")
    if baseline_rate is None:
        raise SystemExit("No 'baseline' condition found — ablate.py run "
                         "incomplete for this model.")

    rows = []
    for condition in rates.index:
        if condition == "baseline":
            continue
        rows.append({
            "layer": int(condition),
            "refusal_rate": rates[condition],
            "baseline_refusal_rate": baseline_rate,
            "drop": baseline_rate - rates[condition],
            "n": n[condition],
        })
    out = pd.DataFrame(rows).sort_values("drop", ascending=False)

    print(f"Baseline refusal rate: {baseline_rate:.2%} (n={n['baseline']})\n")
    print(out.to_string(index=False,
                        formatters={
                            "refusal_rate": "{:.2%}".format,
                            "baseline_refusal_rate": "{:.2%}".format,
                            "drop": "{:+.2%}".format,
                        }))

    out_path = Path(args.results_dir) / f"ablation_summary__{slug}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
