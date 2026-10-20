from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mlxscribe.filters import FilterOptions, is_high_compression, normalize_duplicate_text

SAMPLE_RATE = 16000
SNAP_WINDOW_SECONDS = 15.0
ENERGY_FRAME_SECONDS = 0.1
ENERGY_SMOOTH_FRAMES = 5
RETRY_TEMPERATURE_STEP = 0.2
MAX_RETRY_TEMPERATURE = 1.0


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float


@dataclass(frozen=True)
class QualityOptions:
    # Tuned on 45- and 120-minute drama samples: short interjections ("ん", "ああ") repeat
    # 4-6 times in real dialogue, and a lone looping segment rarely improves on a chunk
    # retry, while retries reliably fixed runs of 8+ and clusters of 5+ repetitive segments.
    max_repeat_run: int = 7
    high_compression_ratio: float = FilterOptions.high_compression_ratio
    max_high_compression_segments: int = 4
    min_coverage: float = 0.3
    speech_rms: float = 0.01
    min_active_share: float = 0.05


def plan_chunks(duration_seconds: float | None, chunk_minutes: float | None) -> list[Chunk]:
    if not duration_seconds or not chunk_minutes:
        return []
    if chunk_minutes <= 0:
        raise ValueError("chunk minutes must be greater than 0")

    chunk_seconds = chunk_minutes * 60
    chunks = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        chunks.append(Chunk(index=index, start=start, end=end))
        start = end
        index += 1
    return chunks


def snap_chunks_to_quiet(
    chunks: list[Chunk],
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: float = SNAP_WINDOW_SECONDS,
) -> list[Chunk]:
    """Move each inner chunk boundary to the quietest nearby point in the audio."""
    if len(chunks) < 2:
        return chunks

    duration = len(audio) / sample_rate
    shortest = min(chunk.end - chunk.start for chunk in chunks[:-1])
    window = min(window_seconds, shortest / 4)
    snapped = []
    start = chunks[0].start
    for chunk in chunks[:-1]:
        low = max(chunk.end - window, start + window)
        high = min(chunk.end + window, duration)
        end = find_quietest_point(audio, sample_rate, low, high, target=chunk.end)
        snapped.append(replace(chunk, start=start, end=end))
        start = end
    snapped.append(replace(chunks[-1], start=start))
    return snapped


def find_quietest_point(
    audio: np.ndarray,
    sample_rate: int,
    low: float,
    high: float,
    target: float,
) -> float:
    frame = int(sample_rate * ENERGY_FRAME_SECONDS)
    first = int(low * sample_rate)
    frame_count = (int(high * sample_rate) - first) // frame
    if frame_count < 1:
        return target

    region = audio[first : first + frame_count * frame].astype(np.float32)
    rms = np.sqrt(np.mean(region.reshape(frame_count, frame) ** 2, axis=1))
    kernel = np.ones(ENERGY_SMOOTH_FRAMES) / ENERGY_SMOOTH_FRAMES
    half = ENERGY_SMOOTH_FRAMES // 2
    energy = np.convolve(np.pad(rms, half, mode="edge"), kernel, mode="valid")
    times = low + (np.arange(frame_count) + 0.5) * ENERGY_FRAME_SECONDS
    # Quietest frame wins; among equally quiet frames, the one nearest the target.
    best = np.lexsort((np.abs(times - target), energy))[0]
    return round(float(times[best]), 2)


def slice_audio(audio: np.ndarray, chunk: Chunk, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return audio[int(chunk.start * sample_rate) : int(chunk.end * sample_rate)]


def offset_result(result: dict, offset: float) -> dict:
    """Shift chunk-relative segment and word timestamps to absolute time."""
    segments = []
    for segment in result.get("segments") or []:
        shifted = {
            **segment,
            "start": float(segment.get("start", 0)) + offset,
            "end": float(segment.get("end", 0)) + offset,
        }
        if segment.get("words"):
            shifted["words"] = [
                {
                    **word,
                    "start": float(word.get("start", 0)) + offset,
                    "end": float(word.get("end", 0)) + offset,
                }
                for word in segment["words"]
            ]
        segments.append(shifted)
    return {**result, "segments": segments}


def merge_chunk_results(results: list[dict], chunks: list[Chunk]) -> dict:
    if not results:
        return {"text": "", "segments": [], "language": None, "chunks": []}

    segments = []
    chunk_summaries = []
    language = next((result.get("language") for result in results if result.get("language")), None)

    for result, chunk in zip(results, chunks, strict=True):
        chunk_segments = result.get("segments") or []
        chunk_summaries.append(
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "segments": len(chunk_segments),
                "language": result.get("language"),
            }
        )
        for segment in chunk_segments:
            segments.append({**segment, "chunk_index": chunk.index})

    segments.sort(key=lambda segment: (float(segment.get("start", 0)), float(segment.get("end", 0))))
    segments = [{**segment, "id": index} for index, segment in enumerate(segments)]
    return {
        "text": "".join(str(segment.get("text", "")) for segment in segments),
        "segments": segments,
        "language": language,
        "chunks": chunk_summaries,
    }


def assess_chunk(
    result: dict,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    options: QualityOptions = QualityOptions(),
) -> dict:
    """Flag a chunk-relative result that looks like a decoding loop or a dropout."""
    segments = result.get("segments") or []
    duration = len(audio) / sample_rate
    active_share = active_audio_share(audio, sample_rate, options.speech_rms)
    has_audio = active_share >= options.min_active_share
    compression_options = FilterOptions(high_compression_ratio=options.high_compression_ratio)
    ratios = [
        float(segment["compression_ratio"])
        for segment in segments
        if isinstance(segment.get("compression_ratio"), (int, float))
    ]
    metrics = {
        "segments": len(segments),
        "max_repeat_run": max_repeat_run(segments),
        "repeat_segments": repeat_run_segments(segments, options.max_repeat_run + 1),
        "max_compression_ratio": round(max(ratios), 3) if ratios else None,
        "high_compression_segments": sum(
            is_high_compression(segment, compression_options) for segment in segments
        ),
        "coverage": round(segment_coverage(segments, duration), 3),
        "active_share": round(active_share, 3),
    }

    flags = []
    if not segments:
        if has_audio:
            flags.append("no_segments")
    else:
        if metrics["max_repeat_run"] > options.max_repeat_run:
            flags.append("repeated_text")
        if metrics["high_compression_segments"] > options.max_high_compression_segments:
            flags.append("high_compression")
        if has_audio and metrics["coverage"] < options.min_coverage:
            flags.append("low_coverage")
    return {"flags": flags, "metrics": metrics}


def quality_rank(assessment: dict) -> tuple:
    """Sort key for attempts: no dropout first, then fewest looping segments, then coverage."""
    flags = assessment["flags"]
    metrics = assessment["metrics"]
    return (
        "no_segments" in flags,
        "low_coverage" in flags,
        metrics["repeat_segments"] + metrics["high_compression_segments"],
        -metrics["coverage"],
    )


def retry_temperatures(retry: int) -> tuple[float, ...]:
    """Temperature fallback for the nth retry: (0.2, 0.4), then (0.4, 0.6), ..."""
    first = min(RETRY_TEMPERATURE_STEP * retry, MAX_RETRY_TEMPERATURE)
    second = min(first + RETRY_TEMPERATURE_STEP, MAX_RETRY_TEMPERATURE)
    return tuple(sorted({round(first, 2), round(second, 2)}))


def max_repeat_run(segments: list[dict]) -> int:
    longest = 0
    run = 0
    previous = None
    for segment in segments:
        text = normalize_duplicate_text(str(segment.get("text", "")))
        run = run + 1 if text and text == previous else 1
        previous = text
        if text:
            longest = max(longest, run)
    return longest


def repeat_run_segments(segments: list[dict], min_run: int) -> int:
    """Count segments that sit in runs of at least min_run identical texts."""
    total = 0
    run = 0
    previous = None
    for segment in [*segments, {"text": ""}]:
        text = normalize_duplicate_text(str(segment.get("text", "")))
        if text and text == previous:
            run += 1
            continue
        if run >= min_run:
            total += run
        run = 1 if text else 0
        previous = text
    return total


def segment_coverage(segments: list[dict], duration: float) -> float:
    if duration <= 0:
        return 0.0
    spans = sorted(
        (
            max(0.0, float(segment.get("start", 0) or 0)),
            min(duration, float(segment.get("end", 0) or 0)),
        )
        for segment in segments
    )
    covered = 0.0
    cursor = 0.0
    for start, end in spans:
        start = max(start, cursor)
        if end > start:
            covered += end - start
            cursor = end
    return covered / duration


def active_audio_share(audio: np.ndarray, sample_rate: int, threshold: float) -> float:
    frame = int(sample_rate * ENERGY_FRAME_SECONDS)
    frame_count = len(audio) // frame
    if frame_count < 1:
        return 0.0
    frames = audio[: frame_count * frame].astype(np.float32).reshape(frame_count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return float(np.mean(rms >= threshold))
