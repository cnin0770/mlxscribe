from __future__ import annotations

import json
from pathlib import Path


def load_transcript(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"transcript JSON must contain a segments list: {path}")
    return data


def transcript_stem(path: Path, source_language: str | None) -> str:
    stem = path.stem
    if source_language and stem.endswith(f".{source_language}"):
        return stem[: -(len(source_language) + 1)]
    return stem


def clean_segments(data: dict) -> list[dict]:
    segments = []
    for segment in data.get("segments") or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "id": segment.get("id"),
                "start": float(segment.get("start", 0)),
                "end": float(segment.get("end", 0)),
                "text": text,
            }
        )
    return segments
