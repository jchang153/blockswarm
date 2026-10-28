"""Episode log: one JSON object per line. This is the ground truth for analysis and replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

LOG_VERSION = 4


class Recorder:
    """Collects records in memory and optionally streams them to a JSONL file."""

    def __init__(self, path: str | Path | None = None):
        self.records: list[dict] = []
        self._fh: IO[str] | None = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.records.append(record)
        if self._fh:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
