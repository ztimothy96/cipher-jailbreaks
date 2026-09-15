"""Shared binary (harmful/harmless) labeled prompt-set loading, used by the
cipher-decode-check and probe-generalization scripts.

Every prompt here is a binary classification example (label = harmful),
unlike measure_refusal_gap.py where only harmful prompts are dispatched.
Reuses common/data.py's CSV loaders so every script can point at the same
HarmBench/AdvBench-style source files.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledRequest:
    id: int
    text: str
    label: int  # 1 = harmful, 0 = harmless


def load_probe_dataset(harmful_csv: str,
                        harmless_csv: str,
                        harmful_col: str = "goal",
                        harmless_col: str = "prompt",
                        max_per_class: int | None = None) -> list[LabeledRequest]:
    from common.data import load_harmful_csv, load_harmless_csv

    harmful = load_harmful_csv(harmful_csv, column=harmful_col)
    harmless = load_harmless_csv(harmless_csv, column=harmless_col)
    if max_per_class is not None:
        harmful = harmful[:max_per_class]
        harmless = harmless[:max_per_class]

    requests = [LabeledRequest(id=i, text=t, label=1) for i, t in enumerate(harmful)]
    requests += [LabeledRequest(id=i, text=t, label=0) for i, t in enumerate(harmless)]
    return requests


def load_smoke_test_dataset(path: str | Path) -> list[LabeledRequest]:
    """Pipeline-plumbing check only — `harmful_placeholder` is explicitly
    NOT real harmful content (see data/smoke_test_prompts.json), so probe
    accuracy on this set is meaningless. Use only to confirm the extraction/
    translation/probe code runs end-to-end without crashing."""
    with open(path) as f:
        data = json.load(f)
    requests = [
        LabeledRequest(id=i, text=t, label=1)
        for i, t in enumerate(data["harmful_placeholder"])
    ]
    requests += [
        LabeledRequest(id=i, text=t, label=0)
        for i, t in enumerate(data["harmless"])
    ]
    return requests
