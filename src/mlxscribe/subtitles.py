"""Subtitle cleanup applied before SRT and TXT export.

Whisper segments are tuned for decoding, not for reading: some run for 15
seconds, some flash for a tenth of a second, and none are wrapped. This pass
turns them into cues that fit on screen, stay up long enough to read, and never
overlap. JSON output keeps the raw segments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

STRONG_BREAK = set("。！？!?.…♪")
WEAK_BREAK = set("、，,;；:：")
NO_LINE_START = set(
    "。！？!?.…、，,;；:：)）」』】〉》〕］]}｝ー〜～"
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ・"
)
NO_LINE_END = set("(（「『【〈《〔［[{｛")
HONORIFIC_PREFIX = set("おご御")
CJK_JOIN_AFTER = STRONG_BREAK | WEAK_BREAK | set("」』）")

# Break-kind penalties: lower is preferred. Distance from the balanced target
# adds 4 points per full line of drift, so a sentence end a few characters off
# centre still beats an arbitrary break in the middle.
BREAK_PENALTY = {
    "strong": 0.0,
    "weak": 0.3,
    "space": 0.6,
    "cjk_phrase": 0.9,  # hiragana into kanji/katakana: usually after a particle
    "cjk": 1.5,
    "cjk_word": 3.0,  # inside a kanji or katakana run, or before okurigana
    "any": 10.0,
}
DRIFT_WEIGHT = 4.0
WORD_GAP_FILL_SECONDS = 0.5


@dataclass(frozen=True)
class SubtitleOptions:
    enabled: bool = True
    max_lines: int = 2
    max_line_chars: int = 42
    max_line_chars_cjk: int = 16
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 7.0
    merge_gap_seconds: float = 0.5
    merge_short_chars: int = 4


def clean_subtitles(result: dict, options: SubtitleOptions) -> tuple[dict, dict]:
    """Return a copy of ``result`` with display-ready segments and a change summary.

    Each output segment has ``start``, ``end``, ``text`` (one logical line, for
    TXT) and ``lines`` (wrapped, for SRT).
    """
    if not options.enabled:
        return result, {"enabled": False}

    stats = {
        "input_segments": 0,
        "dropped_empty": 0,
        "trimmed_to_words": 0,
        "overlaps_fixed": 0,
        "split_segments": 0,
        "split_pieces": 0,
        "split_with_word_timestamps": 0,
        "merged_segments": 0,
        "extended_cues": 0,
        "wrapped_cues": 0,
    }
    cues = normalize_segments(result.get("segments") or [], options, stats)
    cues = fix_overlaps(cues, stats)
    cues = split_cues(cues, options, stats)
    cues = merge_short_cues(cues, options, stats)
    cues = enforce_min_duration(cues, options, stats)

    segments = []
    for index, cue in enumerate(cues):
        lines = wrap_cue_text(cue["text"], options)
        if len(lines) > 1:
            stats["wrapped_cues"] += 1
        segments.append(
            {
                "id": index,
                "start": round(cue["start"], 3),
                "end": round(cue["end"], 3),
                "text": cue["text"],
                "lines": lines,
            }
        )

    summary = {
        "enabled": True,
        **stats,
        "output_cues": len(segments),
        **measure_cues(segments, options),
        "options": asdict(options),
    }
    cleaned = {**result, "segments": segments}
    return cleaned, summary


def measure_cues(segments: list[dict], options: SubtitleOptions) -> dict:
    """Count cues that still break a limit, so the summary shows what cleanup could not fix."""
    long_lines = 0
    too_many_lines = 0
    short = 0
    overlong = 0
    overlapping = 0
    tolerance = 1e-6
    for index, segment in enumerate(segments):
        cap = line_capacity(segment["text"], options)
        long_lines += sum(len(line) > cap for line in segment["lines"])
        too_many_lines += len(segment["lines"]) > options.max_lines
        duration = segment["end"] - segment["start"]
        short += duration < options.min_duration_seconds - tolerance
        overlong += duration > options.max_duration_seconds + tolerance
        if index + 1 < len(segments):
            overlapping += segment["end"] > segments[index + 1]["start"] + tolerance
    return {
        "remaining_long_lines": long_lines,
        "remaining_too_many_lines": too_many_lines,
        "remaining_short_cues": short,
        "remaining_long_cues": overlong,
        "remaining_overlaps": overlapping,
    }


# --- text helpers -----------------------------------------------------------


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3000 <= code <= 0x30FF  # CJK punctuation, hiragana, katakana
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF  # hangul
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def is_cjk_text(text: str) -> bool:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    return sum(is_cjk_char(char) for char in chars) * 2 >= len(chars)


def line_capacity(text: str, options: SubtitleOptions) -> int:
    return options.max_line_chars_cjk if is_cjk_text(text) else options.max_line_chars


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def join_texts(left: str, right: str) -> str:
    if left and right and is_cjk_char(left[-1]) and is_cjk_char(right[0]):
        if left[-1] in CJK_JOIN_AFTER:
            return left + right
    return f"{left} {right}"


def break_kind(text: str, index: int) -> str | None:
    """Classify a break before ``text[index]``; ``None`` means never break here."""
    before, after = text[index - 1], text[index]
    if before.isspace():
        return None  # the break is taken at the space itself
    if after.isspace():
        if before in STRONG_BREAK:
            return "strong"
        if before in WEAK_BREAK:
            return "weak"
        if is_cjk_char(before):
            return "strong"  # a space in CJK text separates utterances
        return "space"
    if after in NO_LINE_START or before in NO_LINE_END:
        return None
    if before in STRONG_BREAK and (is_cjk_char(before) or is_cjk_char(after)):
        return "strong"
    if before in WEAK_BREAK and (is_cjk_char(before) or is_cjk_char(after)):
        return "weak"
    if before in HONORIFIC_PREFIX and script(after) == "kanji":
        return "cjk_word"  # お金, ご飯: the prefix belongs to the next word
    if is_cjk_char(before) or is_cjk_char(after):
        return cjk_break_kind(script(before), script(after))
    return "any"


def script(char: str) -> str:
    code = ord(char)
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF or 0xFF66 <= code <= 0xFF9F:
        return "katakana"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "kanji"
    if char.isalnum():
        return "latin"
    return "other"


def cjk_break_kind(before: str, after: str) -> str:
    if before == "hiragana" and after in ("kanji", "katakana", "latin"):
        return "cjk_phrase"
    if before == after and before in ("kanji", "katakana", "latin"):
        return "cjk_word"
    if before == "kanji" and after == "hiragana":
        return "cjk_word"
    return "cjk"


def split_offsets(
    text: str,
    parts: int,
    cap: int,
    allowed: set[int] | None = None,
    base: int = 0,
) -> list[int]:
    """Choose ``parts - 1`` break offsets that give balanced pieces of at most ``cap`` chars.

    Splits recursively near the balanced position, preferring sentence ends,
    then clause breaks, then spaces, then any CJK character boundary. Pieces
    that cannot fit still get the best available break. ``allowed`` restricts
    breaks to these absolute offsets (word boundaries).
    """
    if parts <= 1 or len(text.strip()) <= 1:
        return []
    left_parts = parts // 2
    right_parts = parts - left_parts
    target = len(text) * left_parts / parts
    best: tuple[tuple[float, float], int] | None = None
    for index in range(1, len(text)):
        if allowed is not None and base + index not in allowed:
            continue
        kind = break_kind(text, index)
        if kind is None:
            continue
        left = text[:index].strip()
        right = text[index:].strip()
        if not left or not right:
            continue
        overflow = max(0, len(left) - left_parts * cap) + max(0, len(right) - right_parts * cap)
        cost = (overflow, BREAK_PENALTY[kind] + abs(index - target) / cap * DRIFT_WEIGHT)
        if best is None or cost < best[0]:
            best = (cost, index)
    if best is None:
        return []
    index = best[1]
    return [
        *split_offsets(text[:index], left_parts, cap, allowed, base),
        base + index,
        *split_offsets(text[index:], right_parts, cap, allowed, base + index),
    ]


def pieces_at(text: str, offsets: list[int]) -> list[str]:
    bounds = [0, *offsets, len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:])]


def wrap_text(text: str, cap: int, max_lines: int) -> list[str] | None:
    """Wrap into at most ``max_lines`` lines of at most ``cap`` chars, or ``None`` if it cannot fit."""
    text = text.strip()
    if len(text) <= cap:
        return [text]
    for count in range(2, max_lines + 1):
        lines = pieces_at(text, split_offsets(text, count, cap))
        if all(len(line) <= cap for line in lines):
            return lines
    return None


def wrap_cue_text(text: str, options: SubtitleOptions) -> list[str]:
    cap = line_capacity(text, options)
    lines = wrap_text(text, cap, options.max_lines)
    if lines is not None:
        return lines
    # Only reachable when a single unbreakable token is longer than the limit;
    # fall back to as many lines as it takes.
    count = math.ceil(len(text) / cap)
    while True:
        lines = pieces_at(text, split_offsets(text, count, cap))
        if all(len(line) <= cap for line in lines) or count > len(text):
            return lines
        count += 1


def fits(text: str, options: SubtitleOptions) -> bool:
    return wrap_text(text, line_capacity(text, options), options.max_lines) is not None


# --- cue passes -------------------------------------------------------------


def normalize_segments(
    segments: list[dict[str, Any]],
    options: SubtitleOptions,
    stats: dict,
) -> list[dict]:
    cues = []
    for segment in segments:
        stats["input_segments"] += 1
        text = normalize_text(segment.get("text", ""))
        if not text:
            stats["dropped_empty"] += 1
            continue
        start = float(segment.get("start", 0) or 0)
        end = max(start, float(segment.get("end", start) or start))
        cue = {"start": start, "end": end, "text": text}
        words = usable_words(segment, text)
        if words:
            cue["words"] = words
            if end - start > options.max_duration_seconds:
                # Whisper often stretches a segment over the silence around
                # it; the words show when the line is actually spoken.
                cue["start"] = max(start, min(end, words[0]["start"]))
                cue["end"] = max(cue["start"], min(end, words[-1]["end"]))
                if (cue["start"], cue["end"]) != (start, end):
                    stats["trimmed_to_words"] += 1
        cues.append(cue)
    cues.sort(key=lambda cue: cue["start"])
    return cues


def usable_words(segment: dict[str, Any], text: str) -> list[dict] | None:
    """Return word timings whose text matches the segment, with whitespace normalized."""
    words = segment.get("words")
    if not isinstance(words, list) or not words:
        return None
    cleaned = []
    for word in words:
        if not isinstance(word, dict):
            return None
        start, end = word.get("start"), word.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        cleaned.append({"word": str(word.get("word", "")), "start": float(start), "end": float(end)})
    if normalize_text("".join(word["word"] for word in cleaned)) != text:
        return None
    return cleaned


def fix_overlaps(cues: list[dict], stats: dict) -> list[dict]:
    for current, following in zip(cues, cues[1:]):
        if current["end"] > following["start"]:
            current["end"] = max(current["start"], following["start"])
            stats["overlaps_fixed"] += 1
    return cues


def split_cues(cues: list[dict], options: SubtitleOptions, stats: dict) -> list[dict]:
    output = []
    for cue in cues:
        pieces = split_cue(cue, options)
        if len(pieces) > 1:
            stats["split_segments"] += 1
            stats["split_pieces"] += len(pieces)
            if "words" in cue:
                stats["split_with_word_timestamps"] += 1
        output.extend(pieces)
    return output


def split_cue(cue: dict, options: SubtitleOptions) -> list[dict]:
    text = cue["text"]
    cap = line_capacity(text, options)
    duration = cue["end"] - cue["start"]
    words = cue.get("words")

    joined, word_offsets = word_layout(words) if words else (text, None)
    allowed = set(word_offsets[1:]) if word_offsets else None

    parts = 1
    if not fits(text, options):
        parts = math.ceil(len(text) / (cap * options.max_lines))
    elif duration > options.max_duration_seconds:
        # A long cue with little text is usually inflated Whisper timing, not
        # several sentences; only split it while each piece keeps half a line.
        by_time = math.ceil(duration / options.max_duration_seconds)
        by_text = max(1, len(text) // max(1, cap // 2))
        if allowed is None:
            # Without word timings the cut's time is a guess, so only cut
            # where the text itself pauses.
            allowed = natural_breaks(joined)
            by_text = min(by_text, len(allowed) + 1)
        parts = min(by_time, by_text)
    if parts <= 1:
        return [cue]

    piece_cap = cap * options.max_lines
    while True:
        offsets = split_offsets(joined, parts, piece_cap, allowed)
        texts = pieces_at(joined, offsets)
        if all(fits(piece, options) for piece in texts) or parts >= len(joined):
            break
        parts += 1
    if len(texts) <= 1:
        return [cue]

    if word_offsets is not None:
        return pieces_from_words(cue, joined, offsets, words, word_offsets)
    return pieces_by_proportion(cue, joined, offsets)


def natural_breaks(text: str) -> set[int]:
    return {
        index
        for index in range(1, len(text))
        if break_kind(text, index) in ("strong", "weak", "space")
    }


def word_layout(words: list[dict]) -> tuple[str, list[int]]:
    """Join words into one string and return each word's start offset in it."""
    text = ""
    offsets = []
    for word in words:
        offsets.append(len(text))
        text += word["word"]
    return text, offsets


def pieces_from_words(
    cue: dict,
    text: str,
    offsets: list[int],
    words: list[dict],
    word_offsets: list[int],
) -> list[dict]:
    bounds = [0, *offsets, len(text)]
    pieces = []
    previous_end = cue["start"]
    for a, b in zip(bounds, bounds[1:]):
        piece_words = [
            word for word, offset in zip(words, word_offsets) if a <= offset < b
        ]
        start = max(previous_end, min(cue["end"], piece_words[0]["start"]))
        end = max(start, min(cue["end"], piece_words[-1]["end"]))
        pieces.append(
            {"start": start, "end": end, "text": normalize_text(text[a:b]), "words": piece_words}
        )
        previous_end = end
    pieces[0]["start"] = cue["start"]
    # A word's end often lands slightly before the next word starts; close
    # small gaps so pieces of one sentence read as continuous.
    for current, following in zip(pieces, pieces[1:]):
        if following["start"] - current["end"] < WORD_GAP_FILL_SECONDS:
            current["end"] = following["start"]
    return pieces


def pieces_by_proportion(cue: dict, text: str, offsets: list[int]) -> list[dict]:
    duration = cue["end"] - cue["start"]
    bounds = [0, *offsets, len(text)]
    times = [cue["start"] + duration * bound / len(text) for bound in bounds]
    times[-1] = cue["end"]
    return [
        {"start": start, "end": end, "text": normalize_text(text[a:b])}
        for (a, b), (start, end) in zip(zip(bounds, bounds[1:]), zip(times, times[1:]))
    ]


def is_short(cue: dict, options: SubtitleOptions) -> bool:
    compact = "".join(char for char in cue["text"] if char.isalnum())
    return (
        cue["end"] - cue["start"] < options.min_duration_seconds
        or len(compact) <= options.merge_short_chars
    )


def merged(left: dict, right: dict) -> dict:
    cue = {
        "start": left["start"],
        "end": max(left["end"], right["end"]),
        "text": join_texts(left["text"], right["text"]),
    }
    if "words" in left and "words" in right:
        cue["words"] = left["words"] + right["words"]
    return cue


def can_merge(left: dict, right: dict, options: SubtitleOptions, max_gap: float) -> bool:
    if right["start"] - left["end"] > max_gap:
        return False
    if max(left["end"], right["end"]) - left["start"] > options.max_duration_seconds:
        return False
    return fits(join_texts(left["text"], right["text"]), options)


def merge_short_cues(cues: list[dict], options: SubtitleOptions, stats: dict) -> list[dict]:
    output: list[dict] = []
    for cue in cues:
        if output:
            previous = output[-1]
            if (is_short(previous, options) or is_short(cue, options)) and can_merge(
                previous, cue, options, options.merge_gap_seconds
            ):
                output[-1] = merged(previous, cue)
                stats["merged_segments"] += 1
                continue
        output.append(cue)
    return output


def enforce_min_duration(cues: list[dict], options: SubtitleOptions, stats: dict) -> list[dict]:
    """Give every cue at least the minimum duration without overlapping its neighbours.

    In order: extend the end into the following gap, extend the start into the
    preceding gap, merge with a neighbour that has room, then borrow time from
    a following cue that is longer than the minimum.
    """
    minimum = options.min_duration_seconds
    index = 0
    while index < len(cues):
        cue = cues[index]
        previous = cues[index - 1] if index > 0 else None
        following = cues[index + 1] if index + 1 < len(cues) else None
        if cue["end"] - cue["start"] >= minimum:
            index += 1
            continue
        stats["extended_cues"] += 1

        limit = following["start"] if following else math.inf
        cue["end"] = max(cue["end"], min(cue["start"] + minimum, limit))
        if cue["end"] - cue["start"] < minimum:
            floor = previous["end"] if previous else 0.0
            cue["start"] = min(cue["start"], max(floor, cue["end"] - minimum))
        if cue["end"] - cue["start"] >= minimum:
            index += 1
            continue

        if following and can_merge(cue, following, options, math.inf):
            cues[index : index + 2] = [merged(cue, following)]
            stats["merged_segments"] += 1
            stats["extended_cues"] -= 1  # re-checked as the merged cue
            continue
        if previous and can_merge(previous, cue, options, math.inf):
            cues[index - 1 : index + 1] = [merged(previous, cue)]
            stats["merged_segments"] += 1
            stats["extended_cues"] -= 1
            index -= 1
            continue

        if following:
            spare_end = following["end"] - minimum
            new_end = min(cue["start"] + minimum, spare_end)
            if new_end > cue["end"]:
                cue["end"] = new_end
                following["start"] = new_end
        index += 1
    return cues
