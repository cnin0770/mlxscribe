from __future__ import annotations

SUPPORTED_FORMATS = frozenset({"json", "srt", "txt"})


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not formats:
        raise ValueError("at least one output format is required")

    unsupported = sorted(set(formats) - SUPPORTED_FORMATS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"unsupported format(s): {', '.join(unsupported)}; supported: {supported}")

    return tuple(dict.fromkeys(formats))
