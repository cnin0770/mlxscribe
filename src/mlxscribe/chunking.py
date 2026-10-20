from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SAMPLE_RATE = 16000
SNAP_WINDOW_SECONDS = 15.0
ENERGY_FRAME_SECONDS = 0.1
ENERGY_SMOOTH_FRAMES = 5


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float


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
