"""Format renderers for the probe-generalization experiment: given a request
string, render it in each test format (language or cipher). Train format is
always plain English.

Ciphers are reused as-is from refusal_gap/ciphers.py (ROT13, Base64,
Leetspeak) — see ALL_CIPHERS there. This module covers only the language
renderers, via the NLLB Translator Modal class in modal_app.py.
"""

import difflib

# FLORES-200 language codes, as used by NLLB (probe_generalization/modal_app.py
# Translator). "english" is the train format; the rest are Phase A test
# formats.
FLORES_CODES = {
    "english": "eng_Latn",
    "chinese": "zho_Hans",
    "japanese": "jpn_Jpan",
    "spanish": "spa_Latn",
}

TEST_LANGUAGES = ["chinese", "japanese", "spanish"]


def translation_round_trip_similarity(original: str,
                                       back_translated: str) -> float:
    """Cheap round-trip similarity heuristic (0-1) for flagging likely-bad
    translations for manual spot-check — analogous to refusal_gap's
    looks_like_decode_noise, not a real semantic-similarity metric. Low
    scores should be spot-checked by hand, not trusted or rejected
    automatically (see docs/probe-generalization-plan.md §5)."""
    return difflib.SequenceMatcher(None, original.strip().lower(),
                                   back_translated.strip().lower()).ratio()
