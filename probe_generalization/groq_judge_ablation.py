"""Rescans an ablate.py output file with Groq's API as the judge.

CAUTION: unlike DeepL translation (which only ever sees request text), this
sends actual elicited completions to a third party — including any
"successful jailbreak" outputs from the ablation sweep. Same
external-API/privacy tradeoff deepl_translate.py already makes for
translation, but a more sensitive category of data; worth being deliberate
about rather than defaulting into it.

Setup: sign up at https://console.groq.com, generate an API key, add
GROQ_API_KEY to .env. Double check the current free-tier model name/rate limits.

Usage:
    python3 probe_generalization/groq_judge_ablation.py \\
        --model Qwen/Qwen2.5-7B-Instruct --judge-model openai/gpt-oss-20b
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.jsonl_output import sort_output_file
from common.modal_infra import model_slug
from probe_generalization.shared.ablation_records import (COMPLETION_FIELD,
                                                          FMT_FIELD,
                                                          REQUEST_FIELD,
                                                          sort_key)
from probe_generalization.shared.judge_prompt import (JUDGE_SYSTEM_PROMPT,
                                                      judge_user_turn,
                                                      parse_label)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_RETRIES = 3


def get_client():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Sign up at https://console.groq.com, "
            "generate a key, then add GROQ_API_KEY=... to .env.")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _call_judge(client, judge_model: str, request: str, completion: str,
                use_reasoning_effort: bool):
    kwargs = dict(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": judge_user_turn(request, completion)
            },
        ],
        max_tokens=200,
        temperature=0,
    )
    if use_reasoning_effort:
        # Groq requires this to be "low"/"medium"/"high" — "none" is not available.
        kwargs["reasoning_effort"] = "low"
    return client.chat.completions.create(**kwargs)


def judge_one(client, judge_model: str, request: str, completion: str) -> str:
    """Returns the judge's raw text, or "" if the model produced no usable
    content — never None, so callers can always call .strip()/.split() on
    the result."""
    last_error = None
    use_reasoning_effort = True
    for attempt in range(MAX_RETRIES):
        try:
            response = _call_judge(client, judge_model, request, completion,
                                   use_reasoning_effort)
        except Exception as e:
            if use_reasoning_effort and "reasoning_effort" in str(e):
                use_reasoning_effort = False
                continue
            last_error = e
            time.sleep(2**attempt)
            continue

        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            print(f"    (empty judge response, finish_reason="
                  f"{finish_reason!r} — treating as UNCLEAR)")
            return ""
        return content
    raise last_error


def load_judged_keys(out_path: Path, fmt_field: str) -> set[tuple]:
    if not out_path.exists():
        return set()
    keys = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["prompt_id"], r["condition"], r[fmt_field]))
    return keys


def judge_track(args, track: str, slug: str):
    fmt_field = FMT_FIELD[track]
    request_field = REQUEST_FIELD[track]
    completion_field = COMPLETION_FIELD[track]

    ablation_path = Path(args.results_dir) / f"ablation_{track}__{slug}.jsonl"
    if not ablation_path.exists():
        print(f"[{track}] no {ablation_path} — skipping (run ablate.py "
              f"first if you expected this track).")
        return

    with open(ablation_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    n_total = len(records)
    if args.max_prompts is not None:
        records = [r for r in records if r["prompt_id"] < args.max_prompts]
    if args.formats is not None:
        wanted = set(args.formats.split(","))
        records = [r for r in records if r[fmt_field] in wanted]
    if args.conditions is not None:
        wanted = set(args.conditions.split(","))
        records = [r for r in records if r["condition"] in wanted]
    if args.even_layers_only:
        records = [
            r for r in records
            if r["condition"] == "baseline" or int(r["condition"]) % 2 == 0
        ]
    if len(records) != n_total:
        print(f"[{track}] subsampled to {len(records)}/{n_total} records "
              f"(--max-prompts={args.max_prompts}, --formats={args.formats}, "
              f"--conditions={args.conditions}, "
              f"--even-layers-only={args.even_layers_only}).")

    out_path = Path(
        args.results_dir) / f"ablation_judged_{track}__{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and out_path.exists():
        out_path.unlink()

    judged = load_judged_keys(out_path, fmt_field)
    if judged:
        print(f"[{track}] resuming: {len(judged)} records already judged, "
              f"skipping.")

    to_judge = [
        r for r in records
        if (r["prompt_id"], r["condition"], r[fmt_field]) not in judged
    ]
    if not to_judge:
        print(f"[{track}] nothing left to judge.")
        sort_output_file(out_path, key=sort_key(fmt_field))
        return

    # One client per worker thread — the openai SDK's HTTP client isn't
    # documented as thread-safe for concurrent .create() calls, and clients
    # are cheap (no connection setup at construction time), so this avoids
    # relying on an unstated guarantee rather than actually saving anything.
    thread_local = threading.local()

    def get_thread_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = get_client()
        return thread_local.client

    def judge_record(r):
        request = r[request_field]
        completion = r[completion_field]
        raw_label = judge_one(get_thread_client(), args.judge_model, request,
                              completion)
        return r, raw_label

    print(f"[{track}] judging {len(to_judge)} completions via Groq "
          f"(judge_model={args.judge_model}, concurrency={args.concurrency}) "
          f"...")

    n_done = 0
    write_lock = threading.Lock()
    with open(out_path, "a") as f, ThreadPoolExecutor(
            max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_record, r): r for r in to_judge}
        for future in as_completed(futures):
            r = futures[future]
            try:
                r, raw_label = future.result()
            except Exception as e:
                print(f"[{track}] [{r['prompt_id']}] {r['condition']:>10s} "
                      f"FAILED after retries: {e!r} — will retry on next run")
                continue

            label = parse_label(raw_label)
            judged_is_refusal = label == "REFUSE"

            out_record = dict(r)
            out_record["judge_raw"] = raw_label
            out_record["judge_label"] = label
            out_record["judge_is_refusal"] = judged_is_refusal

            with write_lock:
                n_done += 1
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{track}] [{n_done}/{len(to_judge)}] "
                  f"[{r['prompt_id']}] {r['condition']:>10s} judge={label:8s}")

    if n_done:
        sort_output_file(out_path, key=sort_key(fmt_field))

    print(f"[{track}] wrote results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        required=True,
                        help="The model being evaluated (identifies which "
                        "ablation_{languages,ciphers}__*.jsonl to read), not "
                        "the judge.")
    parser.add_argument("--judge-model",
                        default="openai/gpt-oss-20b",
                        help="Groq-hosted model id to use as judge — verify "
                        "the current free-tier name at "
                        "https://console.groq.com/docs/models before "
                        "relying on this default.")
    parser.add_argument("--track",
                        choices=["languages", "ciphers"],
                        default=None,
                        help="Judge only this track. Default judges both "
                        "(whichever ablation_{track}__*.jsonl files exist).")
    parser.add_argument("--results-dir",
                        default="results/probe_generalization")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Parallel Groq requests. Calls are independent, so this is "
        "safe to raise; back off if you start seeing 429s (Groq's free "
        "tier has a requests-per-minute cap that varies by model).")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Judge only prompt_id < this many of the harmful prompts."
        "Default judges every prompt in the ablation file.")
    parser.add_argument("--formats",
                        default=None,
                        help="Comma-separated subset (languages or ciphers, "
                        "matching --track) to judge. Default judges every "
                        "format in the ablation file.")
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated subset of conditions to judge (e.g. "
        "'baseline' or 'baseline,0,2'). Default judges every condition in "
        "the ablation file.")
    parser.add_argument(
        "--even-layers-only",
        action="store_true",
        help="Judge only conditions where the ablated layer is even.")
    args = parser.parse_args()

    slug = model_slug(args.model)
    tracks = [args.track] if args.track else ["languages", "ciphers"]
    for track in tracks:
        judge_track(args, track, slug)


if __name__ == "__main__":
    main()
