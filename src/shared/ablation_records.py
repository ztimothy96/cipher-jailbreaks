"""Typed records for ablate.py's two output tracks (languages and ciphers)."""

from dataclasses import asdict, dataclass
from typing import ClassVar

BASELINE = "baseline"

# Field on a record that identifies the format (language or cipher).
FMT_FIELD = {"languages": "language", "ciphers": "cipher"}
REQUEST_FIELD = {"languages": "prompt", "ciphers": "decoded_prompt"}
COMPLETION_FIELD = {"languages": "completion", "ciphers": "decoded_completion"}


def sort_key(fmt_field: str):
    """Groups by format, then by prompt, with baseline ordered before each
    ablated layer — lets a reader scan straight down and see how one
    prompt's completion shifts as the ablated layer changes. Shared by
    ablate.py and groq_judge_ablation.py so both output files use the same
    ordering."""

    def key(r: dict) -> tuple:
        layer = -1 if r["condition"] == BASELINE else int(r["condition"])
        return (r[fmt_field], r["prompt_id"], layer)

    return key


@dataclass
class LanguageAblationRecord:
    TRACK: ClassVar[str] = "languages"

    model: str
    prompt_id: int
    condition: str
    language: str
    prompt: str
    completion: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CipherAblationRecord:
    TRACK: ClassVar[str] = "ciphers"

    model: str
    prompt_id: int
    condition: str
    cipher: str
    encoded_prompt: str
    decoded_prompt: str
    raw_completion: str
    decoded_completion: str

    def to_dict(self) -> dict:
        return asdict(self)
