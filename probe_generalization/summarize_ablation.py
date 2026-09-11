"""Summarizes ablate.py's output into a refusal-rate-per-layer table, ranked
by drop from baseline — the layers most causally load-bearing for refusal
sort to the top.

Can use either the keyword scan (refusal_gap.refusal.is_refusal) or the
LLM-judge label (judge_ablation.py / groq_judge_ablation.py) as the metric
— see --metric. Whichever is used, the choice is never left implicit in the
output: it's a "metric" column in the CSV *and* part of the output
filename (ablation_summary__{model}__{metric}.csv), so a table or plot
found later is self-describing without needing this script's stdout.
--metric auto (the default) picks judge when a judged file exists,
otherwise keyword, and still records which one it picked into both places
— "auto" only affects the choice, never leaves it unrecorded.

Usage:
    python3 probe_generalization/summarize_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct
    python3 probe_generalization/summarize_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct --metric keyword
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
    parser.add_argument("--results-dir",
                        default="results/probe_generalization")
    parser.add_argument(
        "--metric",
        choices=["auto", "keyword", "judge"],
        default="auto",
        help="'auto' uses the judge label if ablation_judged__*.jsonl "
        "exists, else falls back to the keyword scan.")
    args = parser.parse_args()

    slug = model_slug(args.model)
    judged_path = Path(args.results_dir) / f"ablation_judged__{slug}.jsonl"
    keyword_path = Path(args.results_dir) / f"ablation__{slug}.jsonl"

    use_judge = args.metric == "judge" or (args.metric == "auto"
                                           and judged_path.exists())
    path = judged_path if use_judge else keyword_path
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run ablate.py"
            f"{' and judge_ablation.py' if use_judge else ''} first.")
    metric_field = "judge_is_refusal" if use_judge else "is_refusal"
    metric_name = "judge" if use_judge else "keyword"
    print(f"Using metric: {metric_name} ({path})\n")

    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]
    if "language" not in df.columns:
        df["language"] = "english"
    df["language"] = df["language"].fillna("english")

    rows = []
    for lang, lang_df in df.groupby("language"):
        if use_judge:
            disagree_rate = lang_df["keyword_judge_disagree"].mean()
            print(f"[{lang}] keyword/judge disagreement: "
                  f"{disagree_rate:.1%} of {len(lang_df)} records")
        rates = lang_df.groupby("condition")[metric_field].mean()
        n = lang_df.groupby("condition")[metric_field].size()
        baseline_rate = rates.get("baseline")
        if baseline_rate is None:
            print(f"[{lang}] no 'baseline' condition found — skipping "
                  f"(ablate.py run incomplete for this language).")
            continue
        for condition in rates.index:
            if condition == "baseline":
                continue
            rows.append({
                "metric": metric_name,
                "language": lang,
                "layer": int(condition),
                "refusal_rate": rates[condition],
                "baseline_refusal_rate": baseline_rate,
                "drop": baseline_rate - rates[condition],
                "n": n[condition],
            })
    if not rows:
        raise SystemExit("No complete (language, layer) results found.")
    out = pd.DataFrame(rows).sort_values(["language", "drop"],
                                         ascending=[True, False])

    for lang, lang_out in out.groupby("language"):
        baseline_rate = lang_out["baseline_refusal_rate"].iloc[0]
        print(f"[{lang}] baseline refusal rate: {baseline_rate:.2%}\n")
        print(
            lang_out.drop(columns=["language", "metric"]).to_string(
                index=False,
                formatters={
                    "refusal_rate": "{:.2%}".format,
                    "baseline_refusal_rate": "{:.2%}".format,
                    "drop": "{:+.2%}".format,
                }))
        print()

    out_path = Path(
        args.results_dir) / f"ablation_summary__{slug}__{metric_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} (metric={metric_name})")


if __name__ == "__main__":
    main()
