"""Shared (system_prompt, encoded_request) rendering for a given format —
used by both understanding_check.py and extract_activations.py so they
render identical prompts. Split out from understanding_check.py because it's
an @app.local_entrypoint() script: importing it (rather than this module)
would re-register its `main` on the shared Modal app and collide with the
importing script's own entrypoint.
"""

import json
from pathlib import Path

from common.ciphers import ALL_CIPHERS
from common.dataset import LabeledRequest
from probe_generalization.shared.formats import TEST_LANGUAGES

LANGUAGE_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Respond in {language}, the same language "
    "as the user's message.")

CIPHER_NAMES = [name for name in ALL_CIPHERS if name != "plaintext"]
ALL_FORMATS = ["english"] + TEST_LANGUAGES + CIPHER_NAMES

# Which activations__<group> subdirectory (see extract_activations.py) a
# format's saved activations live under.
FORMATS_BY_GROUP = {
    "languages": ["english"] + TEST_LANGUAGES,
    "ciphers": CIPHER_NAMES,
}
FORMAT_GROUP = {
    fmt: group
    for group, fmts in FORMATS_BY_GROUP.items() for fmt in fmts
}


def load_translations(path: Path) -> dict[tuple[int, int, str], str]:
    translations = {}
    if not path.exists():
        return translations
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["prompt_id"], record["label"], record["language"])
            translations[key] = record["translated"]
    return translations


def build_prompt(fmt: str, req: LabeledRequest,
                  translations: dict[tuple[int, int, str], str]) -> tuple[str, str]:
    """Returns (system_prompt, encoded_request) for a given format. Raises
    KeyError if a language translation is missing (run translate_prompts.py
    first)."""
    if fmt == "english":
        return "You are a helpful assistant.", req.text
    if fmt in TEST_LANGUAGES:
        key = (req.id, req.label, fmt)
        if key not in translations:
            raise KeyError(
                f"No translation for prompt_id={req.id} label={req.label} "
                f"language={fmt} — run translate_prompts.py first.")
        return (LANGUAGE_SYSTEM_PROMPT_TEMPLATE.format(
            language=fmt.capitalize()), translations[key])
    cipher = ALL_CIPHERS[fmt]
    return cipher.system_prompt_template, cipher.encode(req.text)


def looks_like_noise(text: str) -> bool:
    """Crude noise filter only, not a real understanding check. A refusal is
    NOT noise (it's coherent evidence the model understood the request), so
    this only catches empty/degenerate output."""
    letters = sum(c.isalpha() for c in text)
    return len(text.strip()) == 0 or letters / max(len(text), 1) < 0.1
