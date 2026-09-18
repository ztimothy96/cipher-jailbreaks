"""Plots AUROC by layer per format from a probe_results__*.csv, saved as a
local PNG.

Usage:
    python3 src/plot_probe_results.py \\
        --model Qwen/Qwen2.5-7B-Instruct
    python3 src/plot_probe_results.py \\
        --model Qwen/Qwen2.5-7B-Instruct --formats english,letter_spaced
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug

# Fixed per-format color, in the validated 8-hue categorical order (see the
# dataviz skill's references/palette.md) — a format keeps the same color
# regardless of which other formats are plotted alongside it ("color follows
# the entity, never its rank"). The 4 languages take the first 4 slots as
# before; letter_spaced (the cipher under active study) takes the 5th so it
# always renders in a validated, CVD-safe hue. The remaining ciphers, which
# are rarely plotted together with 8 others, share the muted fallback.
COLOR_ORDER = [
    "english",
    "chinese",
    "japanese",
    "spanish",
    "letter_spaced",
    "rot13",
    "base64",
    "leetspeak",
]
_VALIDATED_HUES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
COLORS = dict(zip(COLOR_ORDER, _VALIDATED_HUES))
FALLBACK_COLOR = "#888888"  # unassigned formats (pig_latin, atbash, ...)
MAX_SAFE_SERIES = len(COLOR_ORDER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated subset of formats to plot, e.g. "
        "english,letter_spaced. Default: every format in the CSV.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = model_slug(args.model)
    csv_path = Path(args.results_dir) / "tables" / f"probe_results__{slug}.csv"
    df = pd.read_csv(csv_path)

    if args.formats is not None:
        wanted = args.formats.split(",")
        unknown = set(wanted) - set(df["format"].unique())
        if unknown:
            raise SystemExit(f"Format(s) not in {csv_path}: {unknown}. "
                             f"Available: {sorted(df['format'].unique())}")
        df = df[df["format"].isin(wanted)]

    n_formats = df["format"].nunique()
    if n_formats > MAX_SAFE_SERIES:
        print(f"Warning: plotting {n_formats} formats at once, beyond the "
              f"{MAX_SAFE_SERIES} colors validated as distinguishable "
              f"together — some will share the fallback gray. Narrow with "
              f"--formats, or split into multiple plots.")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for fmt, group in df.groupby("format"):
        group = group.sort_values("layer")
        ax.plot(group["layer"], group["auroc"], label=fmt.capitalize(),
               color=COLORS.get(fmt, FALLBACK_COLOR), linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Probe transfer by layer — {args.model}")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = f"__{'_'.join(sorted(df['format'].unique()))}" if args.formats else ""
        out_path = Path(
            args.results_dir) / "figures" / f"probe_results__{slug}{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
