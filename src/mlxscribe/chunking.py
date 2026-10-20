from __future__ import annotations

from dataclasses import dataclass


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


def format_clip_timestamps(chunk: Chunk) -> str:
    return f"{chunk.start:.3f},{chunk.end:.3f}"
