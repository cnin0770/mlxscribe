from __future__ import annotations

from pathlib import Path


def write_srt(path: Path, result: dict) -> None:
    segments = result.get("segments") or []
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(float(segment.get("start", 0)))
        end = format_srt_timestamp(float(segment.get("end", 0)))
        lines = segment.get("lines")
        text = "\n".join(lines) if lines else str(segment.get("text", "")).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")

    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    milliseconds_total = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
