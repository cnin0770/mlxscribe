from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class FilterOptions:
    enabled: bool = False
    mode: str = "conservative"
    leading_window_seconds: float = 15.0
    min_duration_seconds: float = 0.05
    high_no_speech_prob: float = 0.8
    low_avg_logprob: float = -1.0
    high_compression_ratio: float = 2.4
    min_repetition_chars: int = 4
    repetition_ratio: float = 0.65
    duplicate_repeat_limit: int = 2


def filter_transcript(result: dict, options: FilterOptions) -> tuple[dict, dict]:
    if not options.enabled:
        summary = {
            "enabled": False,
            "mode": options.mode,
            "dropped_segments": 0,
            "dropped": [],
            "rules": [],
        }
        return result, summary

    kept_segments = []
    dropped = []
    rules = set()
    previous_normalized_text = None
    consecutive_text_count = 0

    for segment in result.get("segments") or []:
        normalized_text = normalize_duplicate_text(str(segment.get("text", "")))
        if normalized_text and normalized_text == previous_normalized_text:
            consecutive_text_count += 1
        else:
            previous_normalized_text = normalized_text
            consecutive_text_count = 1

        reason = drop_reason(segment, options, consecutive_text_count)
        if reason is None:
            kept_segments.append(segment)
            continue

        rules.add(reason)
        dropped.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "reason": reason,
            }
        )

    filtered = {**result, "segments": renumber_segments(kept_segments)}
    filtered["text"] = "".join(str(segment.get("text", "")) for segment in filtered["segments"])

    summary = {
        "enabled": True,
        "mode": options.mode,
        "dropped_segments": len(dropped),
        "kept_segments": len(filtered["segments"]),
        "dropped": dropped,
        "rules": sorted(rules),
        "options": options.__dict__,
    }
    filtered["filter"] = summary
    return filtered, summary


def drop_reason(
    segment: dict[str, Any],
    options: FilterOptions,
    consecutive_text_count: int = 1,
) -> str | None:
    text = str(segment.get("text", "")).strip()
    start = float(segment.get("start", 0) or 0)
    end = float(segment.get("end", 0) or 0)
    duration = end - start

    if not text:
        return "empty_text"
    if duration <= options.min_duration_seconds:
        return "near_zero_duration"
    if is_low_confidence_silence(segment, options):
        return "low_confidence_silence"
    if start <= options.leading_window_seconds and is_repetitive_text(text, options):
        return "leading_repetition"
    if is_high_compression_repetition(segment, options):
        return "high_compression_repetition"
    if (
        consecutive_text_count > options.duplicate_repeat_limit
        and is_high_compression(segment, options)
    ):
        return "repeated_adjacent_text"
    return None


def is_low_confidence_silence(segment: dict[str, Any], options: FilterOptions) -> bool:
    no_speech_prob = segment.get("no_speech_prob")
    avg_logprob = segment.get("avg_logprob")
    if not isinstance(no_speech_prob, (int, float)):
        return False
    if not isinstance(avg_logprob, (int, float)):
        return False
    return no_speech_prob >= options.high_no_speech_prob and avg_logprob <= options.low_avg_logprob


def is_high_compression_repetition(segment: dict[str, Any], options: FilterOptions) -> bool:
    if not is_high_compression(segment, options):
        return False
    return is_repetitive_text(str(segment.get("text", "")), options)


def is_high_compression(segment: dict[str, Any], options: FilterOptions) -> bool:
    compression_ratio = segment.get("compression_ratio")
    if not isinstance(compression_ratio, (int, float)):
        return False
    return compression_ratio >= options.high_compression_ratio


def is_repetitive_text(text: str, options: FilterOptions) -> bool:
    compact = compact_text(text)
    if len(compact) < options.min_repetition_chars:
        return False

    if len(set(compact)) == 1:
        return True

    char_counts = Counter(compact)
    if char_counts.most_common(1)[0][1] / len(compact) >= options.repetition_ratio:
        return True

    phrases = repeated_phrases(text)
    if phrases:
        phrase_counts = Counter(phrases)
        _, count = phrase_counts.most_common(1)[0]
        if count >= 3 and count / len(phrases) >= options.repetition_ratio:
            return True

    return has_repeated_ngram(compact)


def repeated_phrases(text: str) -> list[str]:
    return [
        part
        for part in re.split(r"[、。，,.!！?？…\s]+", text)
        if len(part) >= 2
    ]


def compact_text(text: str) -> str:
    ignored = set(" \t\r\n、。，,.!！?？…・「」『』（）()[]")
    return "".join(char for char in text if char not in ignored)


def normalize_duplicate_text(text: str) -> str:
    return compact_text(text).lower()


def has_repeated_ngram(text: str) -> bool:
    for size in range(2, min(8, len(text) // 2) + 1):
        chunk = text[:size]
        repeats = len(text) // size
        if chunk * repeats == text[: repeats * size]:
            return True

        chunks = [
            text[index : index + size]
            for index in range(0, len(text) - size + 1, size)
        ]
        if len(chunks) < 3:
            continue
        _, count = Counter(chunks).most_common(1)[0]
        if count >= 3 and (count * size) / len(text) >= 0.65:
            return True
    return False


def renumber_segments(segments: list[dict]) -> list[dict]:
    return [{**segment, "id": index} for index, segment in enumerate(segments)]
