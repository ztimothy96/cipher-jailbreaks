"""Summarizes ablate.py's output into a refusal-rate-per-layer table, ranked
by drop from baseline — the layers most causally load-bearing for refusal
sort to the top. Plots the results as a line chart.

Uses the LLM-judge label from groq_judge_ablation.py as the metric.

Usage:
    python3 src/summarize_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct
    python3 src/summarize_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct --track ciphers
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug
from src.shared.ablation_records import FMT_FIELD
from src.shared.judge_prompt import VALID_LABELS

# Stack order (bottom to top) — REFUSE/COMPLY first since they're the
# meaningful outcomes; ECHO/UNCLEAR (~gibberish) last since they mean the
# judge couldn't classify real intent. Fixed order, not cycled, so a label
# always gets the same color across every breakdown chart.
LABEL_ORDER = ["REFUSE", "COMPLY", "ECHO", "UNCLEAR"]
assert set(LABEL_ORDER) == VALID_LABELS
LABEL_COLORS = {
    "REFUSE": "#2a78d6",
    "COMPLY": "#eb6834",
    "ECHO": "#1baf7a",
    "UNCLEAR": "#9a9a9a",
}


def save_ablation_results(args: argparse.Namespace):
    slug = model_slug(args.model)
    fmt_field = FMT_FIELD[args.track]
    path = Path(
        args.results_dir) / "raw" / f"ablation_judged_{args.track}__{slug}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run ablate.py and groq_judge_ablation.py first."
        )
    metric_field = "judge_is_refusal"
    metric_name = "judge"
    print(f"Using metric: {metric_name} ({path})\n")

    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]
    if args.max_prompts is not None:
        n_before = len(df)
        df = df[df["prompt_id"] < args.max_prompts]
        print(f"Restricted to prompt_id < {args.max_prompts}: "
              f"{len(df)}/{n_before} records (keeps language curves "
              f"comparable when some languages are judged on more "
              f"prompts than others).\n")
    if args.conditions is not None:
        wanted = set(args.conditions.split(","))
        df = df[df["condition"].isin(wanted)]

    # echo_rate is reported alongside refusal_rate for every track, but only
    # the ciphers track actually sees ECHO fire much (a model echoing/
    # decoding-back a cipher instead of complying or refusing) — languages
    # summaries just carry a near-zero column.
    rows = []
    for fmt, fmt_df in df.groupby(fmt_field):
        rates = fmt_df.groupby("condition")[metric_field].mean()
        echo_rates = fmt_df.groupby("condition")["judge_label"].apply(
            lambda labels: (labels == "ECHO").mean())
        n = fmt_df.groupby("condition")[metric_field].size()
        baseline_rate = rates.get("baseline")
        if baseline_rate is None:
            print(f"[{fmt}] no 'baseline' condition found — skipping "
                  f"(ablate.py run incomplete for this format).")
            continue
        for condition in rates.index:
            if condition == "baseline":
                continue
            rows.append({
                "metric": metric_name,
                "format": fmt,
                "layer": int(condition),
                "refusal_rate": rates[condition],
                "baseline_refusal_rate": baseline_rate,
                "drop": baseline_rate - rates[condition],
                "echo_rate": echo_rates[condition],
                "n": n[condition],
            })
    if not rows:
        print("No ablated-layer results to summarize (e.g. a baseline-only "
              "run) — skipping the refusal-rate-by-layer chart.")
        return
    out = pd.DataFrame(rows).sort_values(["format", "drop"],
                                         ascending=[True, False])

    for fmt, fmt_out in out.groupby("format"):
        baseline_rate = fmt_out["baseline_refusal_rate"].iloc[0]
        print(f"[{fmt}] baseline refusal rate: {baseline_rate:.2%}\n")
        print(
            fmt_out.drop(columns=["format", "metric"]).to_string(
                index=False,
                formatters={
                    "refusal_rate": "{:.2%}".format,
                    "baseline_refusal_rate": "{:.2%}".format,
                    "drop": "{:+.2%}".format,
                    "echo_rate": "{:.2%}".format,
                }))
        print()

    out_path = Path(
        args.results_dir
    ) / "tables" / f"ablation_summary__{slug}__{metric_name}__{args.track}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} (metric={metric_name})")


def plot_ablation_results(args: argparse.Namespace):
    slug = model_slug(args.model)
    metric_name = "judge"
    csv_path = Path(
        args.results_dir
    ) / "tables" / f"ablation_summary__{slug}__{metric_name}__{args.track}.csv"
    if not csv_path.exists():
        print(f"No {csv_path} — skipping the refusal-rate-by-layer chart.")
        return
    df = pd.read_csv(csv_path)
    if args.even_layers_only:
        df = df[df["layer"] % 2 == 0]

    # Only the languages track has a fixed, small, meaningful set of formats
    # worth stable colors for — ciphers fall back to the default color cycle.
    COLORS = {
        "english": "#2a78d6",
        "chinese": "#eb6834",
        "japanese": "#1baf7a",
        "spanish": "#eda100",
    }
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for fmt, group in df.groupby("format"):
        group = group.sort_values("layer")
        color = COLORS.get(fmt)
        line, = ax.plot(group["layer"],
                        group["refusal_rate"],
                        label=f"{fmt.capitalize()} (ablated)",
                        color=color,
                        linewidth=2,
                        marker="o",
                        markersize=4)
        ax.axhline(group["baseline_refusal_rate"].iloc[0],
                   color=line.get_color(),
                   linewidth=1,
                   linestyle="--",
                   alpha=0.6)

    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Refusal rate")
    ax.set_title(f"Refusal rate under single-layer ablation — {args.model}\n"
                 f"track={args.track} metric={metric_name} (dashed = "
                 f"unablated baseline, same color per format)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    suffix = "__even_layers" if args.even_layers_only else ""
    out_path = Path(
        args.results_dir
    ) / "figures" / f"ablation_summary__{slug}__{metric_name}__{args.track}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path} (metric={metric_name}, "
          f"even_layers_only={args.even_layers_only})")


def save_label_breakdown(args: argparse.Namespace):
    """Unlike save_ablation_results (refusal rate only), this keeps the full
    REFUSE/COMPLY/ECHO/UNCLEAR split per (format, condition) — refusal rate
    alone can't distinguish a model that's complying from one that's just
    echoing the cipher back or producing gibberish."""
    slug = model_slug(args.model)
    fmt_field = FMT_FIELD[args.track]
    path = Path(
        args.results_dir) / "raw" / f"ablation_judged_{args.track}__{slug}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run ablate.py and groq_judge_ablation.py first."
        )

    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]
    if args.max_prompts is not None:
        df = df[df["prompt_id"] < args.max_prompts]
    if args.conditions is not None:
        wanted = set(args.conditions.split(","))
        df = df[df["condition"].isin(wanted)]
    if args.even_layers_only:
        # pandas | isn't short-circuited like Python's `or` — astype(int)
        # would run on the "baseline" rows too before the OR is applied, so
        # coerce non-numeric conditions to NaN instead (NaN % 2 == 0 is
        # False, and those rows are already kept by the baseline mask).
        is_baseline = df["condition"] == "baseline"
        is_even = pd.to_numeric(df["condition"], errors="coerce") % 2 == 0
        df = df[is_baseline | is_even]

    rows = []
    for (fmt, condition), group in df.groupby([fmt_field, "condition"]):
        layer = -1 if condition == "baseline" else int(condition)
        n = len(group)
        counts = group["judge_label"].value_counts()
        row = {"format": fmt, "condition": condition, "layer": layer, "n": n}
        for label in LABEL_ORDER:
            row[label.lower()] = counts.get(label, 0) / n
        rows.append(row)
    if not rows:
        raise SystemExit("No judged records found.")
    out = pd.DataFrame(rows).sort_values(["format", "layer"])

    out_path = Path(
        args.results_dir
    ) / "tables" / f"ablation_label_breakdown__{slug}__{args.track}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_label_breakdown(args: argparse.Namespace):
    slug = model_slug(args.model)
    csv_path = Path(
        args.results_dir
    ) / "tables" / f"ablation_label_breakdown__{slug}__{args.track}.csv"
    df = pd.read_csv(csv_path)

    formats = sorted(df["format"].unique())
    fig, axes = plt.subplots(1,
                             len(formats),
                             figsize=(5.5 * len(formats), 5),
                             dpi=150,
                             squeeze=False)
    axes = axes[0]

    for ax, fmt in zip(axes, formats):
        fmt_df = df[df["format"] == fmt].sort_values("layer")
        x = range(len(fmt_df))
        x_labels = [
            "baseline" if layer == -1 else str(layer)
            for layer in fmt_df["layer"]
        ]
        bottom = [0.0] * len(fmt_df)
        for label in LABEL_ORDER:
            values = fmt_df[label.lower()].tolist()
            ax.bar(x,
                  values,
                  bottom=bottom,
                  width=0.7,
                  color=LABEL_COLORS[label],
                  label=label.capitalize())
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels,
                           rotation=90 if len(x_labels) > 12 else 0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Ablated layer")
        ax.set_title(fmt.capitalize())
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Share of responses")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle(f"Judge label breakdown under single-layer ablation — "
                f"{args.model}\ntrack={args.track}")
    fig.tight_layout()

    suffix = "__even_layers" if args.even_layers_only else ""
    out_path = Path(
        args.results_dir
    ) / "figures" / f"ablation_label_breakdown__{slug}__{args.track}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--track",
                        choices=["languages", "ciphers"],
                        default="languages")
    parser.add_argument("--results-dir",
                        default="results")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Restrict to prompt_id < this many, so formats judged on "
        "different numbers of prompts are still compared on the same "
        "shared subset.")
    parser.add_argument("--even-layers-only",
                        action="store_true",
                        help="Plot only even-numbered ablated layers.")
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated subset of conditions to summarize (e.g. "
        "'baseline' or 'baseline,0,2'). Default summarizes every condition "
        "in the judged file.")
    args = parser.parse_args()
    save_ablation_results(args)
    plot_ablation_results(args)
    save_label_breakdown(args)
    plot_label_breakdown(args)


if __name__ == "__main__":
    main()
