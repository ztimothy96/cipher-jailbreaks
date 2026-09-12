"""Step 1 behavioral baseline, run on a remote GPU via Modal.

Setup (one-time):
    pip install -r requirements.txt
    modal setup          # interactive browser login

Usage:
    modal run refusal_gap/measure_refusal_gap.py --smoke-test
    modal run refusal_gap/measure_refusal_gap.py --model Qwen/Qwen2.5-14B-Instruct --smoke-test
    modal run refusal_gap/measure_refusal_gap.py --harmful-csv path/to/advbench.csv --harmful-col goal
    modal run refusal_gap/measure_refusal_gap.py --harmful-csv path/to/advbench.csv \\
        --ciphers pig_latin,leetspeak,letter_spaced,middle_swap,snakespeak
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refusal_gap.ciphers import ALL_CIPHERS, build_prompt
from refusal_gap.data import load_harmful_csv, load_smoke_test_prompts
from refusal_gap.modal_app import ChatModel
from common.modal_infra import DEFAULT_MODEL, app
from refusal_gap.refusal import is_refusal


def looks_like_decode_noise(text: str) -> bool:
    """Crude noise filter only — not a real decode-quality check. Per the
    user's plan, decode quality for the smoke test is checked by manual
    inspection of decoded_completion in the output file, not automatically."""
    letters = sum(c.isalpha() for c in text)
    return len(text.strip()) == 0 or letters / max(len(text), 1) < 0.3


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_completed(out_path: Path, model_name: str) -> set[tuple[int, str]]:
    """Only counts a (prompt_id, cipher) pair as done if the existing record
    was produced by the same model — guards against accidentally resuming
    into a file that (via an explicit --out) mixes models."""
    if not out_path.exists():
        return set()
    completed = set()
    mismatched = 0
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("model") != model_name:
                mismatched += 1
                continue
            completed.add((record["prompt_id"], record["cipher"]))
    if mismatched:
        print(
            f"Warning: {mismatched} record(s) in {out_path} belong to a "
            f"different model than '{model_name}' and were ignored for "
            f"resume purposes. Consider using separate output files per model."
        )
    return completed


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    harmful_csv: str = None,
    harmful_col: str = "goal",
    max_prompts: int = 20,
    max_new_tokens: int = 256,
    model: str = DEFAULT_MODEL,
    out: str = None,
    resume: bool = True,
    ciphers: str = None,
):
    if ciphers is not None:
        wanted = ciphers.split(",")
        unknown = set(wanted) - set(ALL_CIPHERS)
        if unknown:
            raise SystemExit(f"Unknown cipher(s): {unknown}. Valid: "
                             f"{sorted(ALL_CIPHERS)}")
        selected_ciphers = {name: ALL_CIPHERS[name] for name in wanted}
    else:
        selected_ciphers = ALL_CIPHERS

    if smoke_test:
        prompts = load_smoke_test_prompts(
            Path(__file__).resolve().parent.parent / "data" /
            "smoke_test_prompts.json")
        harmful_prompts = prompts[
            "harmless"]  # smoke test only, not actually harmful
    elif harmful_csv:
        harmful_prompts = load_harmful_csv(harmful_csv, column=harmful_col)
    else:
        raise SystemExit("Pass either --smoke-test or --harmful-csv <path>.")

    harmful_prompts = harmful_prompts[:max_prompts]

    if out is None:
        out = f"results/refusal_gap/refusal_gap__{model_slug(model)}.jsonl"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
        out_path.unlink()
    completed = load_completed(out_path, model)
    if completed:
        print(
            f"Resuming: {len(completed)} (prompt, cipher) pairs already done "
            f"for model '{model}', will be skipped.")

    # Build the full (prompt, cipher) index, then drop anything already done.
    index = []
    for i, request in enumerate(harmful_prompts):
        for cipher_name, cipher in selected_ciphers.items():
            if (i, cipher_name) in completed:
                continue
            system_prompt, encoded_request = build_prompt(cipher, request)
            index.append((i, cipher_name, system_prompt, encoded_request))

    if not index:
        print(
            "Nothing left to do — all requested (prompt, cipher) pairs are already in the output file."
        )
        return

    print(
        f"Dispatching {len(index)} generations to Modal (model={model}) via .map() ..."
    )
    chat_model = ChatModel(model_name=model)

    system_prompts = [row[2] for row in index]
    user_turns = [row[3] for row in index]
    max_new_tokens_list = [max_new_tokens] * len(index)

    with open(out_path, "a") as f:
        results = chat_model.generate.map(
            system_prompts,
            user_turns,
            max_new_tokens_list,
            order_outputs=True,
            return_exceptions=True,
        )

        for (prompt_id, cipher_name, system_prompt,
             encoded_request), completion in zip(index, results):
            if isinstance(completion, Exception):
                print(
                    f"[{prompt_id}] {cipher_name:10s} FAILED after retries: {completion!r} — will retry on next --resume run"
                )
                continue  # not written, so a re-run with --resume will pick it back up

            cipher = ALL_CIPHERS[cipher_name]
            try:
                decoded_completion = cipher.decode(
                    completion) if cipher_name != "plaintext" else completion
            except Exception:
                decoded_completion = "<decode_error>"

            record = {
                "model":
                model,
                "prompt_id":
                prompt_id,
                "cipher":
                cipher_name,
                "system_prompt":
                system_prompt,
                "encoded_request":
                encoded_request,
                "raw_completion":
                completion,
                "decoded_completion":
                decoded_completion,
                "decode_looks_valid":
                not looks_like_decode_noise(decoded_completion),
                "is_refusal":
                is_refusal(decoded_completion),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush(
            )  # durability: survive a crash/dropped connection right after this line
            print(
                f"[{prompt_id}] {cipher_name:10s} refusal={record['is_refusal']!s:5} decode_ok={record['decode_looks_valid']!s:5}"
            )

    print(f"\nWrote results to {out_path}")
