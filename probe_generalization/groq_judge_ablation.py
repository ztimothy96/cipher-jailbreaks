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

from probe_generalization.shared.judge_prompt import (JUDGE_SYSTEM_PROMPT,
                                                    judge_user_turn, parse_label)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_RETRIES = 3


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


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


def load_judged_keys(out_path: Path) -> set[tuple]:
    if not out_path.exists():
        return set()
    keys = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["prompt_id"], r["condition"], r["language"]))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        required=True,
                        help="The model being evaluated (identifies which "
                        "ablation__*.jsonl to read), not the judge.")
    parser.add_argument("--judge-model",
                        default="openai/gpt-oss-20b",
                        help="Groq-hosted model id to use as judge — verify "
                        "the current free-tier name at "
                        "https://console.groq.com/docs/models before "
                        "relying on this default.")
    parser.add_argument("--ablation-path", default=None)
    parser.add_argument("--out", default=None)
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
    parser.add_argument("--languages",
                        default=None,
                        help="Comma-separated subset to judge."
                        "Default judges every language in the ablation file.")
    parser.add_argument(
        "--even-layers-only",
        action="store_true",
        help="Judge only conditions where the ablated layer is even.")
    args = parser.parse_args()

    slug = model_slug(args.model)
    ablation_path = Path(
        args.ablation_path
        or f"results/probe_generalization/ablation__{slug}.jsonl")
    if not ablation_path.exists():
        raise SystemExit(f"Missing {ablation_path} — run ablate.py first.")

    records = []
    with open(ablation_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    for r in records:
        r.setdefault("language", "english")

    n_total = len(records)
    if args.max_prompts is not None:
        records = [r for r in records if r["prompt_id"] < args.max_prompts]
    if args.languages is not None:
        wanted = set(args.languages.split(","))
        records = [r for r in records if r["language"] in wanted]
    if args.even_layers_only:
        # None of these filters depend on `language`, so every language
        # present in `records` gets exactly the same (prompt_id, condition)
        # slice — required for the per-language curves to be comparable
        # rather than differing by sampling noise.
        records = [
            r for r in records
            if r["condition"] == "baseline" or int(r["condition"]) % 2 == 0
        ]
    if len(records) != n_total:
        print(f"Subsampled to {len(records)}/{n_total} records "
              f"(--max-prompts={args.max_prompts}, "
              f"--languages={args.languages}, "
              f"--even-layers-only={args.even_layers_only}).")

    out_path = Path(
        args.out
        or f"results/probe_generalization/ablation_judged__{slug}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and out_path.exists():
        out_path.unlink()

    judged = load_judged_keys(out_path)
    if judged:
        print(f"Resuming: {len(judged)} records already judged, skipping.")

    to_judge = [
        r for r in records
        if (r["prompt_id"], r["condition"], r["language"]) not in judged
    ]
    if not to_judge:
        print("Nothing left to judge.")
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
        raw_label = judge_one(get_thread_client(), args.judge_model,
                              r["prompt"], r["completion"])
        return r, raw_label

    print(f"Judging {len(to_judge)} completions via Groq "
          f"(judge_model={args.judge_model}, concurrency={args.concurrency}) "
          f"...")

    n_disagree = 0
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
                print(f"[{r['prompt_id']}] {r['condition']:>10s} FAILED "
                      f"after retries: {e!r} — will retry on next run")
                continue

            label = parse_label(raw_label)
            judged_is_refusal = label == "REFUSE"
            disagrees = judged_is_refusal != r["is_refusal"]

            out_record = dict(r)
            out_record["judge_raw"] = raw_label
            out_record["judge_label"] = label
            out_record["judge_is_refusal"] = judged_is_refusal
            out_record["keyword_judge_disagree"] = disagrees

            with write_lock:
                n_disagree += disagrees
                n_done += 1
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f.flush()
            flag = " DISAGREE" if disagrees else ""
            print(f"[{n_done}/{len(to_judge)}] [{r['prompt_id']}] "
                  f"{r['condition']:>10s} keyword={r['is_refusal']!s:5} "
                  f"judge={label:8s}{flag}")

    if n_done:
        sort_output_file(out_path)

    print(f"\nWrote results to {out_path}")
    if n_done:
        print(f"Keyword/judge disagreement: {n_disagree}/{n_done} "
              f"({n_disagree / n_done:.1%}) in this batch")


def sort_output_file(out_path: Path):

    def sort_key(r):
        condition = r["condition"]
        layer = -1 if condition == "baseline" else int(condition)
        return (r["language"], layer, r["prompt_id"])

    with open(out_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=sort_key)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
