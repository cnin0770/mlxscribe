from __future__ import annotations

import json
from pathlib import Path


def load_benchmarks(paths: list[Path]) -> list[dict]:
    benchmarks = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        benchmarks.append(data)
    return benchmarks


def print_benchmark_summary(paths: list[Path]) -> None:
    benchmarks = load_benchmarks(paths)
    if not benchmarks:
        print("No benchmark files found.")
        return

    rows = []
    for item in benchmarks:
        phases = item.get("phase_timings") or {}
        rows.append(
            [
                Path(item.get("input", item["_path"])).name,
                format_seconds(item.get("source_duration_seconds")),
                format_seconds(item.get("elapsed_seconds")),
                format_factor(item.get("realtime_factor")),
                format_seconds(phases.get("extract_audio_seconds")),
                format_seconds(phases.get("transcribe_seconds")),
                item.get("model", ""),
            ]
        )

    headers = ["input", "duration", "total", "speed", "extract", "transcribe", "model"]
    print_table(headers, rows)


def format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value >= 60:
        minutes = int(value // 60)
        seconds = value - minutes * 60
        return f"{minutes}m{seconds:04.1f}s"
    return f"{value:.2f}s"


def format_factor(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.2f}x"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
