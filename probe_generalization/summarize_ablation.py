"""Summarizes ablate.py's output into a refusal-rate-per-layer table, ranked
by drop from baseline — the layers most causally load-bearing for refusal
sort to the top. Plots the results as a line chart.

Can use either the keyword scan (common.refusal.is_refusal) or the
LLM-judge label (groq_judge_ablation.py) as the metric.

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

import matplotlib.pyplot as plt


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def save_ablation_results(args: argparse.Namespace):
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
    if args.max_prompts is not None:
        n_before = len(df)
        df = df[df["prompt_id"] < args.max_prompts]
        print(f"Restricted to prompt_id < {args.max_prompts}: "
              f"{len(df)}/{n_before} records (keeps language curves "
              f"comparable when some languages are judged on more "
              f"prompts than others).\n")

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


def plot_ablation_results(args: argparse.Namespace):
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
    if args.even_layers_only:
        df = df[df["layer"] % 2 == 0]

    COLORS = {
        "english": "#2a78d6",
        "chinese": "#eb6834",
        "japanese": "#1baf7a",
        "spanish": "#eda100",
    }
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for lang, group in df.groupby("language"):
        group = group.sort_values("layer")
        color = COLORS.get(lang, "#888888")
        ax.plot(group["layer"],
                group["refusal_rate"],
                label=f"{lang.capitalize()} (ablated)",
                color=color,
                linewidth=2,
                marker="o",
                markersize=4)
        ax.axhline(group["baseline_refusal_rate"].iloc[0],
                   color=color,
                   linewidth=1,
                   linestyle="--",
                   alpha=0.6)

    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Refusal rate")
    ax.set_title(f"Refusal rate under single-layer ablation — {args.model}\n"
                 f"metric={metric_name} (dashed = unablated baseline, "
                 f"same color per language)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    suffix = "__even_layers" if args.even_layers_only else ""
    out_path = Path(
        args.results_dir
    ) / f"ablation_summary__{slug}__{metric_name}{suffix}.png"
    fig.savefig(out_path)
    print(f"Saved {out_path} (metric={metric_name}, "
          f"even_layers_only={args.even_layers_only})")


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
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Restrict to prompt_id < this many, so languages judged on "
        "different numbers of prompts are still compared on the same "
        "shared subset.")
    parser.add_argument(
        "--even-layers-only",
        action="store_true",
        help="Plot only even-numbered ablated layers.")
    args = parser.parse_args()

    save_ablation_results(args)
    plot_ablation_results(args)


if __name__ == "__main__":
    main()
