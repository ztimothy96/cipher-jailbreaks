"""Prompt set loading.

This module does NOT bundle a harmful-request benchmark. Per the research
plan, harmful prompts should come from an existing red-teaming benchmark
(HarmBench, AdvBench) rather than being authored for this project. Download
one of those separately and point --harmful_csv at it.

Expected CSV schema: a single column containing the request text. AdvBench's
`harmful_behaviors.csv` uses a `goal` column; HarmBench exports typically use
`behavior`. Pass --harmful_col to match whichever you're using.
"""

import json
from pathlib import Path

import pandas as pd


def load_smoke_test_prompts(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        data = json.load(f)
    return {
        "harmless": data["harmless"],
        "harmful": data["harmful_placeholder"]
    }


def load_harmful_csv(path: str | Path, column: str = "goal") -> list[str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {path}. Available columns: "
            f"{list(df.columns)}. Pass --harmful_col to match your CSV's schema."
        )
    return df[column].dropna().astype(str).tolist()


def load_harmless_csv(path: str | Path, column: str = "prompt") -> list[str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {path}. Available columns: "
            f"{list(df.columns)}.")
    return df[column].dropna().astype(str).tolist()
