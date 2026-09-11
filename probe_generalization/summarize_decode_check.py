"""Prints decode_score distribution per cipher from cipher_decode_check.py's
output, to calibrate a decode_ok threshold by looking at real scores rather
than guessing one (see conversation — Leetspeak's ceiling may legitimately
differ from ROT13/Base64's).

Usage:
    python3 probe_generalization/summarize_decode_check.py \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Single sequential hue (blue, mid-dark step) — these are all the same
# measure (decode_score) grouped by category, not distinct series, so one
# hue for every box avoids implying identity differences that aren't there.
_BOX_COLOR = "#2a78d6"
_MEDIAN_COLOR = "#0b0b0b"
_GRID_COLOR = "#e1e0d9"
_AXIS_COLOR = "#c3c2b7"
_MUTED_TEXT = "#898781"
_PRIMARY_TEXT = "#0b0b0b"


def plot_decode_scores(df: pd.DataFrame, out_path: Path) -> None:
    order = (df.groupby("cipher")["decode_score"].median().sort_values(
        ascending=True).index.tolist())
    data = [df.loc[df["cipher"] == c, "decode_score"].values for c in order]

    fig_height = 0.6 * len(order) + 1.2
    fig, ax = plt.subplots(figsize=(7, fig_height))

    bp = ax.boxplot(data,
                    vert=False,
                    tick_labels=order,
                    patch_artist=True,
                    widths=0.6,
                    medianprops={"color": _MEDIAN_COLOR, "linewidth": 2},
                    whiskerprops={"color": _AXIS_COLOR},
                    capprops={"color": _AXIS_COLOR},
                    flierprops={
                        "markeredgecolor": _AXIS_COLOR,
                        "markersize": 4,
                    })
    for box in bp["boxes"]:
        box.set_facecolor(_BOX_COLOR)
        box.set_edgecolor(_BOX_COLOR)
        box.set_alpha(0.85)

    for i, c in enumerate(order):
        median = df.loc[df["cipher"] == c, "decode_score"].median()
        ax.text(median,
                i + 1 + 0.38,
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_PRIMARY_TEXT)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("decode_score (higher = closer to original text)",
                  color=_MUTED_TEXT)
    ax.set_title("Cipher decodability", color=_PRIMARY_TEXT, loc="left")
    ax.tick_params(colors=_PRIMARY_TEXT)
    ax.xaxis.grid(True, color=_GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS_COLOR)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results/probe_generalization")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Read the _smoketest output file — pass this if you ran "
        "cipher_decode_check.py with --smoke-test, same filename suffix "
        "convention as that script.")
    parser.add_argument(
        "--path", default=None,
        help="Explicit path override, if neither --model's default nor "
        "--smoke-test's suffix match your output file's actual name.")
    parser.add_argument(
        "--show-worst", type=int, default=3,
        help="Print this many lowest-scoring (original, completion) pairs "
        "per cipher, to see what a bad decode actually looks like.")
    parser.add_argument(
        "--plot", nargs="?", const="", default=None,
        help="Also save a box plot comparing decode_score across ciphers. "
        "Optionally pass a path; default is "
        "'<results-dir>/cipher_decode_check<suffix>__<model>.png'.")
    args = parser.parse_args()

    slug = model_slug(args.model)
    if args.path:
        path = Path(args.path)
    else:
        suffix = "_smoketest" if args.smoke_test else ""
        path = Path(
            args.results_dir) / f"cipher_decode_check{suffix}__{slug}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run cipher_decode_check.py first "
            f"(pass --smoke-test here too if that's how you ran it).")
    records = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["model"] == args.model]

    for cipher, group in df.groupby("cipher"):
        scores = group["decode_score"]
        percentiles = np.percentile(scores, [10, 25, 50, 75, 90])
        print(f"[{cipher}] n={len(group)}  "
              f"mean={scores.mean():.3f}  "
              f"p10={percentiles[0]:.3f} p25={percentiles[1]:.3f} "
              f"p50={percentiles[2]:.3f} p75={percentiles[3]:.3f} "
              f"p90={percentiles[4]:.3f}")
        if args.show_worst:
            worst = group.nsmallest(args.show_worst, "decode_score")
            for _, r in worst.iterrows():
                print(f"    score={r['decode_score']:.3f} "
                      f"original={r['original']!r}")
                print(f"      completion={r['completion']!r}")
        print()

    if args.plot is not None:
        if args.plot:
            plot_path = Path(args.plot)
        else:
            plot_path = path.with_suffix(".png")
        plot_decode_scores(df, plot_path)
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
