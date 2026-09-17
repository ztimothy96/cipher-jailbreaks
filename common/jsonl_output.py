"""Shared helpers for the append-only JSONL result files written by the
Modal entrypoints (ablate.py, cipher_decode_check.py, ...)."""

import json
from pathlib import Path
from typing import Callable


def sort_output_file(out_path: Path, key: Callable[[dict], tuple]) -> None:
    """Rewrites out_path with rows ordered by `key`, so results from
    different resumed runs — which append in whatever order they were
    dispatched — line up consistently. Writes to a temp file and renames,
    so a crash mid-sort can't leave out_path truncated/corrupt."""
    if not out_path.exists():
        return
    with open(out_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=key)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
