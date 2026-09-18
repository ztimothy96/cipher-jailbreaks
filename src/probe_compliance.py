"""Evaluates a saved English-trained probe against model compliance
and plots the result.

Answers: "does the probe direction, applied zero-shot to this format's
activations, predict whether the model complied or refused?" To keep
results comparable to probe.py, the primary metric is refusal_auroc:
label=1 for REFUSE. Only COMPLY/REFUSE judged rows feed it. ECHO and
UNCLEAR are excluded, though ECHO can still be plotted as a third class.

Activations are keyed by prompt_id, but that id is reused across the
paired harmful/benign prompt with different `labels` entries. Evaluation
matches only the harmful row per prompt_id, since compliance is only
judged for harmful prompts; --harmless-reference/its per-panel
include_benign_as_comply pulls in the paired benign row separately (see
below).

Each panel's x-axis/legend framing depends on how it was evaluated:
  - Plain eval: judged model behavior on harmful prompts (Refuse/Comply,
    plus Echo if --include-echo).
  - A format used as --harmless-reference: also pulls each judged prompt_id's
    paired BENIGN activation under an assumed COMPLY label. Framing becomes 
    Harmful/Harmless (prompt ground truth) rather than Refuse/Comply (judged
    behavior); rows get source='assumed' vs 'judged' in the scores CSV. This 
    covers formats with no judged COMPLY at all (e.g. plaintext English): 
    refusal_auroc reduces to the harmful/benign-midpoint AUROC by construction.

Shaded bands are the interquartile range (25th-75th percentile) due to the 
small size and outliers of the COMPLY class.

Usage:
    python3 src/probe_compliance.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats ciphers:letter_spaced
    python3 src/probe_compliance.py --model Qwen/Qwen2.5-7B-Instruct \\
        --formats languages:english,ciphers:letter_spaced \\
        --harmless-reference languages:english --include-echo
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.modal_infra import model_slug
from src.shared.ablation_records import BASELINE, FMT_FIELD
from src.shared.prompt_rendering import FORMAT_GROUP

VALID_JUDGE_LABELS = {"REFUSE", "COMPLY"}

# Validated hues (see plot_probe_results.py's COLOR_ORDER/palette.md) — fixed
# per role so colors never collide across the up-to-4 series a panel can
# show at once (Refuse/Comply/Echo + the cross-format harmless reference).
CLASS_ORDER = ["REFUSE", "COMPLY", "ECHO"]
CLASS_COLORS = {"REFUSE": "#2a78d6", "COMPLY": "#eb6834", "ECHO": "#eda100"}
REFERENCE_COLOR = "#1baf7a"
CATEGORY_DISPLAY = {
    "judged": {
        "REFUSE": "Refuse",
        "COMPLY": "Comply",
        "ECHO": "Echo"
    },
    "assumed": {
        "REFUSE": "Harmful",
        "COMPLY": "Harmless"
    },
}


def load_judge_labels(results_dir: Path, model: str, track: str, fmt: str,
                      conditions: set[str],
                      wanted_labels: set[str]) -> dict[int, str]:
    """prompt_id -> judge_label, restricted to wanted_labels (e.g. {"REFUSE",
    "COMPLY"} or {"ECHO"})."""
    slug = model_slug(model)
    path = results_dir / f"ablation_judged_{track}__{slug}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run ablate.py and groq_judge_ablation.py first."
        )
    fmt_field = FMT_FIELD[track]
    pid_to_label = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r["model"] != model or r[fmt_field] != fmt
                or r["condition"] not in conditions
                or r["judge_label"] not in wanted_labels):
            continue
        pid_to_label[r["prompt_id"]] = r["judge_label"]
    return pid_to_label


def load_activations(activations_root: Path, fmt: str, model: str) -> dict:
    path = activations_root / f"activations__{FORMAT_GROUP[fmt]}" / model_slug(
        model) / f"{fmt}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path} — run extract_activations.py first.")
    data = np.load(path)
    return {
        "activations": data["activations"],
        "labels": data["labels"],
        "prompt_ids": data["prompt_ids"],
    }


def evaluate(model: str,
             track: str,
             fmt: str,
             conditions: set[str],
             activations_root: Path,
             probes_root: Path,
             results_dir: Path,
             out_dir: Path,
             include_benign_as_comply: bool = False,
             include_echo: bool = False,
             write: bool = True,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full scoring pipeline for one (track, fmt) and returns
    (per-layer AUROC df, per-example scores df). Writes both to CSV under
    out_dir unless write=False (see the module docstring for the flags'
    effect on the resulting categories/framing)."""

    def log(*a):
        if verbose:
            print(*a)

    pid_to_judge_label = load_judge_labels(results_dir, model, track, fmt,
                                           conditions, VALID_JUDGE_LABELS)
    if not pid_to_judge_label:
        raise SystemExit(f"No COMPLY/REFUSE-judged rows found for "
                         f"format={fmt} conditions={conditions}.")
    n_comply = sum(1 for v in pid_to_judge_label.values() if v == "COMPLY")
    n_refuse = sum(1 for v in pid_to_judge_label.values() if v == "REFUSE")
    log(f"[{track}:{fmt}] Judged rows: {len(pid_to_judge_label)} "
        f"(comply={n_comply}, refuse={n_refuse})")

    acts = load_activations(activations_root, fmt, model)
    activations, harm_labels, prompt_ids = (acts["activations"],
                                            acts["labels"], acts["prompt_ids"])
    wanted = [(pid, 1, label, int(label == "REFUSE"), "judged")
              for pid, label in pid_to_judge_label.items()]
    if include_echo:
        pid_to_echo = load_judge_labels(results_dir, model, track, fmt,
                                        conditions, {"ECHO"})
        wanted += [(pid, 1, "ECHO", None, "judged") for pid in pid_to_echo]
        log(f"[{track}:{fmt}] Echo rows: {len(pid_to_echo)}")
    if include_benign_as_comply:
        wanted += [(pid, 0, "COMPLY", 0, "assumed")
                   for pid in pid_to_judge_label]
    wanted_lookup = {
        (pid, harm): (judge_label, refusal_target, source)
        for pid, harm, judge_label, refusal_target, source in wanted
    }

    mask = np.array([(pid, harm) in wanted_lookup
                     for pid, harm in zip(prompt_ids, harm_labels)])
    X = activations[mask]
    matched_keys = list(
        zip(prompt_ids[mask].tolist(), harm_labels[mask].tolist()))
    if len(set(matched_keys)) != len(matched_keys):
        raise SystemExit(
            "Duplicate (prompt_id, label) after masking — unexpected, "
            "activation extraction may have changed.")
    judge_labels = [wanted_lookup[k][0] for k in matched_keys]
    refusal_targets = [wanted_lookup[k][1] for k in matched_keys]
    sources = [wanted_lookup[k][2] for k in matched_keys]
    auroc_mask = np.array([t is not None for t in refusal_targets])
    y = np.array([t for t in refusal_targets if t is not None])
    log(f"[{track}:{fmt}] Matched activations: n={len(matched_keys)} "
        f"({auroc_mask.sum()} used for refusal_auroc)")

    probes_path = probes_root / f"{model_slug(model)}.npz"
    if not probes_path.exists():
        raise SystemExit(f"Missing {probes_path} — run probe.py first.")
    probe = np.load(probes_path)
    weights, biases = probe["weights"], probe["biases"]
    n_layers = weights.shape[0]

    conditions_str = ",".join(sorted(conditions))
    matched_pids = [k[0] for k in matched_keys]
    rows = []
    score_rows = []
    for layer in range(n_layers):
        scores = X[:, layer, :] @ weights[layer] + biases[layer]
        refusal_auroc = roc_auc_score(y, scores[auroc_mask])
        rows.append({
            "model": model,
            "track": track,
            "format": fmt,
            "conditions": conditions_str,
            "layer": layer,
            "refusal_auroc": refusal_auroc,
            "comply_auroc": 1 - refusal_auroc,
            "n": len(y),
        })
        for pid, judge_label, score, source in zip(matched_pids, judge_labels,
                                                   scores, sources):
            score_rows.append({
                "model": model,
                "track": track,
                "format": fmt,
                "conditions": conditions_str,
                "layer": layer,
                "prompt_id": int(pid),
                "judge_label": judge_label,
                "source": source,
                "score": float(score),
            })

    df = pd.DataFrame(rows)
    scores_df = pd.DataFrame(score_rows)
    best = df.loc[df["refusal_auroc"].idxmax()]
    log(f"[{track}:{fmt}] Best layer for predicting refusal: "
        f"{int(best['layer'])} (refusal_auroc={best['refusal_auroc']:.3f})")

    if write:
        suffix = "".join([
            "__with_assumed_benign" if include_benign_as_comply else "",
            "__with_echo" if include_echo else "",
        ])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"probe_compliance__{model_slug(model)}__{track}__{fmt}"
            f"{suffix}.csv")
        df.to_csv(out_path, index=False)
        log(f"Saved {out_path}")

        scores_path = out_dir / (
            f"probe_compliance_scores__{model_slug(model)}__{track}__{fmt}"
            f"{suffix}.csv")
        scores_df.to_csv(scores_path, index=False)
        log(f"Saved {scores_path}")

    return df, scores_df


def layer_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("layer")["score"].agg(median="median",
                                            q25=lambda s: s.quantile(0.25),
                                            q75=lambda s: s.quantile(0.75),
                                            n="size").sort_index()


def build_figure(
        model: str,
        panels: list[tuple[str, str, pd.DataFrame]],
        reference: tuple[str, pd.DataFrame] | None = None) -> plt.Figure:
    """panels: [(track, fmt, scores_df), ...], one per subplot.
    reference: (fmt, scores_df) whose 'assumed' Harmless class is overlaid
    on every judged-framing panel — scores_df must have 'source'=='assumed'
    rows (see evaluate()'s include_benign_as_comply)."""
    reference_fmt, reference_stats = None, None
    if reference is not None:
        reference_fmt, ref_df = reference
        if not ("source" in ref_df.columns and
                (ref_df["source"] == "assumed").any()):
            raise SystemExit(f"harmless-reference format '{reference_fmt}' "
                             f"has no 'assumed' Harmless class — rerun its "
                             f"evaluate() with include_benign_as_comply=True.")
        reference_stats = layer_stats(
            ref_df[ref_df["judge_label"] == "COMPLY"])

    fig, axes = plt.subplots(1,
                             len(panels),
                             figsize=(5.5 * len(panels), 4.5),
                             dpi=150,
                             squeeze=False)
    axes = axes[0]

    for ax, (_track, fmt, df) in zip(axes, panels):
        framing = "assumed" if (
            "source" in df.columns and
            (df["source"] == "assumed").any()) else "judged"
        display = CATEGORY_DISPLAY[framing]

        present = [c for c in CLASS_ORDER if c in df["judge_label"].unique()]
        for judge_label in present:
            stats = layer_stats(df[df["judge_label"] == judge_label])
            color = CLASS_COLORS[judge_label]
            n = int(stats["n"].iloc[0])
            ax.plot(stats.index,
                    stats["median"],
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=f"{display[judge_label]} (n={n})")
            ax.fill_between(stats.index,
                            stats["q25"],
                            stats["q75"],
                            color=color,
                            alpha=0.32,
                            linewidth=0)

        if framing == "judged" and reference_stats is not None:
            n_ref = int(reference_stats["n"].iloc[0])
            ax.plot(reference_stats.index,
                    reference_stats["median"],
                    color=REFERENCE_COLOR,
                    linewidth=2,
                    linestyle="--",
                    marker="^",
                    markersize=4,
                    label=f"{reference_fmt} harmless, reference (n={n_ref})")
            ax.fill_between(reference_stats.index,
                            reference_stats["q25"],
                            reference_stats["q75"],
                            color=REFERENCE_COLOR,
                            alpha=0.28,
                            linewidth=0)

        ax.axhline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_title(f"{fmt}" + (" (harmful/harmless prompt)" if framing ==
                                 "assumed" else " (refuse/comply)"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[0].set_ylabel("Score along refusal direction\n(median, IQR band)")
    fig.suptitle(f"Median probe score by layer — {model}\n"
                 f"baseline (no ablation); dashed line = probe's "
                 f"harmful/benign decision boundary")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--formats",
        required=True,
        help="Comma-separated track:format pairs to score and plot as "
        "panels, e.g. 'languages:english,ciphers:letter_spaced'.")
    parser.add_argument(
        "--harmless-reference",
        default=None,
        help="track:format pair to overlay as a reference line on every "
        "judged-framing (Refuse/Comply) panel — automatically evaluated "
        "with the assumed-benign-as-Harmless class it needs (see module "
        "docstring), whether or not it's also in --formats, "
        "e.g. 'languages:english'.")
    parser.add_argument(
        "--conditions",
        default=BASELINE,
        help="Comma-separated ablate.py conditions to evaluate against "
        f"(default: '{BASELINE}', i.e. no ablation).")
    parser.add_argument(
        "--include-echo",
        action="store_true",
        help="Also score ECHO-judged examples for every judged-framing "
        "format, plotted as a third class.")
    parser.add_argument("--activations-root", default="results/src")
    parser.add_argument("--probes-root", default="results/src/probes")
    parser.add_argument("--results-dir", default="results/src")
    parser.add_argument("--out-dir", default="results/src")
    parser.add_argument("--out", default=None, help="Plot output path.")
    args = parser.parse_args()

    conditions = set(args.conditions.split(","))
    activations_root = Path(args.activations_root)
    probes_root = Path(args.probes_root)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    format_specs = [tuple(spec.split(":")) for spec in args.formats.split(",")]
    reference_spec = (tuple(args.harmless_reference.split(":"))
                      if args.harmless_reference else None)

    evaluated = {}  # (track, fmt) -> scores_df

    def eval_once(track: str, fmt: str, include_benign_as_comply: bool):
        key = (track, fmt)
        if key not in evaluated:
            _, scores_df = evaluate(
                args.model,
                track,
                fmt,
                conditions,
                activations_root,
                probes_root,
                results_dir,
                out_dir,
                include_benign_as_comply=include_benign_as_comply,
                include_echo=args.include_echo)
            evaluated[key] = scores_df
        return evaluated[key]

    panels = []
    for track, fmt in format_specs:
        include_benign = (track, fmt) == reference_spec
        panels.append((track, fmt, eval_once(track, fmt, include_benign)))

    reference = None
    if reference_spec:
        ref_track, ref_fmt = reference_spec
        reference = (ref_fmt, eval_once(ref_track, ref_fmt, True))

    fig = build_figure(args.model, panels, reference)

    if args.out:
        out_path = Path(args.out)
    else:
        slug = model_slug(args.model)
        fmt_slug = "_".join(fmt for _, fmt in format_specs)
        out_path = results_dir / f"probe_compliance_trend__{slug}__{fmt_slug}.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
