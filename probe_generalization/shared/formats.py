"""Format renderers for the probe-generalization experiment: given a request
string, render it in each test format (language or cipher). Train format is
always plain English.

Ciphers are reused as-is from common/ciphers.py (ROT13, Base64,
Leetspeak) — see ALL_CIPHERS there. This module covers only the language
renderers, via the DeepL API (probe_generalization/deepl_translate.py).

Translation-quality QA is manual review (sample_for_review.py).
"""

# DeepL's language codes (probe_generalization/deepl_translate.py).
# "english" is the train format; the rest are Phase A test formats.
LANGUAGE_CODES = {
    "english": "EN",
    "chinese": "ZH",
    "japanese": "JA",
    "spanish": "ES",
}

TEST_LANGUAGES = ["chinese", "japanese", "spanish"]
