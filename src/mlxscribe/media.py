from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required but was not found on PATH")
    return path


def probe_duration(input_path: Path) -> float | None:
    ffprobe = require_tool("ffprobe")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    duration = data.get("format", {}).get("duration")
    return float(duration) if duration is not None else None


def extract_audio(input_path: Path, output_path: Path) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
