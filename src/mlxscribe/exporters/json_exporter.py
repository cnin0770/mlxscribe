from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
