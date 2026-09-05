"""DeepL API translation — no GPU needed, so this runs locally.

Setup: sign up at https://www.deepl.com/pro-api, generate an API key, copy
.env.example to .env, and fill in DEEPL_API_KEY there. .env is gitignored —
never commit it or hardcode the key in source.
"""

import os

import deepl
from dotenv import load_dotenv

load_dotenv()

# DeepL accepts a list per request; chunk to stay well under documented
# per-request limits rather than tune this precisely against undocumented
# server-side behavior.
BATCH_SIZE = 50


def get_translator() -> deepl.Translator:
    api_key = os.environ.get("DEEPL_API_KEY")
    if not api_key:
        raise SystemExit(
            "DEEPL_API_KEY is not set. Sign up at https://www.deepl.com/pro-api, "
            "generate a key, then copy .env.example to .env and fill it in.")
    return deepl.Translator(api_key)


def batch_translate(translator: deepl.Translator, texts: list[str],
                    source_lang: str,
                    target_lang: str) -> list[str | Exception]:
    """Translates texts in chunks of BATCH_SIZE, preserving order. A failed
    chunk yields the same exception for every text in that chunk (caller can
    retry via --resume, same pattern as the Modal .map() calls elsewhere in
    this repo) rather than failing the whole run."""
    results: list[str | Exception] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        try:
            translated = translator.translate_text(chunk,
                                                   source_lang=source_lang,
                                                   target_lang=target_lang)
            results.extend(r.text for r in translated)
        except Exception as e:
            results.extend([e] * len(chunk))
    return results
