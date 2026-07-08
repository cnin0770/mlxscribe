from __future__ import annotations

from pathlib import Path

from mlxscribe.exporters.srt import format_srt_timestamp


def write_txt(path: Path, result: dict) -> None:
    segments = result.get("segments") or []
    if segments:
        lines = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = format_srt_timestamp(float(segment.get("start", 0))).replace(",", ".")
            lines.append(f"[{start}] {text}")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return

    text = str(result.get("text", "")).strip()
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")
