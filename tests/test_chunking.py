import numpy as np

from mlxscribe.chunking import (
    SAMPLE_RATE,
    Chunk,
    assess_chunk,
    merge_chunk_results,
    offset_result,
    plan_chunks,
    quality_rank,
    retry_temperatures,
    slice_audio,
    snap_chunks_to_quiet,
)


def tone(seconds: float) -> np.ndarray:
    samples = np.arange(int(seconds * SAMPLE_RATE))
    return (0.5 * np.sin(2 * np.pi * 220 * samples / SAMPLE_RATE)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_plan_chunks_splits_duration_by_minutes() -> None:
    chunks = plan_chunks(125.0, 1.0)

    assert chunks == [
        Chunk(index=0, start=0.0, end=60.0),
        Chunk(index=1, start=60.0, end=120.0),
        Chunk(index=2, start=120.0, end=125.0),
    ]


def test_plan_chunks_disabled_without_duration_or_minutes() -> None:
    assert plan_chunks(None, 10) == []
    assert plan_chunks(100, None) == []


def test_snap_moves_boundary_into_nearby_silence() -> None:
    # Speech everywhere except a 1s pause at 66-67s; nominal boundary at 60s.
    audio = np.concatenate([tone(66), silence(1), tone(53)])
    chunks = snap_chunks_to_quiet(plan_chunks(120.0, 1.0), audio)

    assert len(chunks) == 2
    assert 66.0 <= chunks[0].end <= 67.0
    assert chunks[1].start == chunks[0].end
    assert chunks[0].start == 0.0
    assert chunks[1].end == 120.0


def test_snap_prefers_pause_nearest_planned_boundary() -> None:
    audio = np.concatenate([tone(52), silence(1), tone(9), silence(1), tone(57)])
    chunks = snap_chunks_to_quiet(plan_chunks(120.0, 1.0), audio)

    assert 62.0 <= chunks[0].end <= 63.0


def test_snap_keeps_boundary_within_window() -> None:
    # The only pause is outside the window, so constant speech ties everywhere and the
    # boundary stays at the planned point rather than drifting to the window edge.
    audio = np.concatenate([tone(100), silence(1), tone(19)])
    chunks = snap_chunks_to_quiet(plan_chunks(120.0, 1.0), audio, window_seconds=10)

    assert abs(chunks[0].end - 60.0) <= 0.5


def test_snap_leaves_single_chunk_alone() -> None:
    chunks = plan_chunks(30.0, 1.0)
    assert snap_chunks_to_quiet(chunks, silence(30)) == chunks


def test_slice_audio_uses_chunk_bounds() -> None:
    audio = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
    sliced = slice_audio(audio, Chunk(index=1, start=2.0, end=3.5))

    assert len(sliced) == int(1.5 * SAMPLE_RATE)
    assert sliced[0] == 2 * SAMPLE_RATE


def test_offset_result_shifts_segments_and_words() -> None:
    result = offset_result(
        {
            "segments": [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hi",
                    "words": [{"word": "hi", "start": 1.0, "end": 1.5}],
                }
            ]
        },
        60.0,
    )

    segment = result["segments"][0]
    assert (segment["start"], segment["end"]) == (61.0, 62.0)
    assert (segment["words"][0]["start"], segment["words"][0]["end"]) == (61.0, 61.5)


def test_merge_chunk_results_sorts_and_renumbers_segments() -> None:
    merged = merge_chunk_results(
        [
            {
                "language": "ja",
                "segments": [
                    {"id": 0, "start": 10.0, "end": 11.0, "text": "first"},
                ],
            },
            {
                "language": "ja",
                "segments": [
                    {"id": 0, "start": 65.0, "end": 66.0, "text": "second"},
                ],
            },
        ],
        [
            Chunk(index=0, start=0.0, end=60.0),
            Chunk(index=1, start=60.0, end=120.0),
        ],
    )

    assert [segment["id"] for segment in merged["segments"]] == [0, 1]
    assert [segment["text"] for segment in merged["segments"]] == ["first", "second"]
    assert [segment["chunk_index"] for segment in merged["segments"]] == [0, 1]
    assert merged["text"] == "firstsecond"
    assert merged["language"] == "ja"


def segment(start: float, end: float, text: str, compression_ratio: float = 1.2) -> dict:
    return {"start": start, "end": end, "text": text, "compression_ratio": compression_ratio}


def test_assess_chunk_passes_clean_speech() -> None:
    result = {"segments": [segment(0.0, 4.0, "こんにちは"), segment(4.0, 9.0, "元気ですか")]}

    assessment = assess_chunk(result, tone(10))

    assert assessment["flags"] == []
    assert assessment["metrics"]["coverage"] == 0.9
    assert assessment["metrics"]["max_repeat_run"] == 1


def test_assess_chunk_flags_repeated_text_loop() -> None:
    segments = [segment(index, index + 1, "ありがとう。") for index in range(6)]

    assessment = assess_chunk({"segments": segments}, tone(6))

    assert "repeated_text" in assessment["flags"]
    assert assessment["metrics"]["max_repeat_run"] == 6


def test_assess_chunk_flags_high_compression() -> None:
    result = {"segments": [segment(0.0, 9.0, "ああああああ", compression_ratio=3.1)]}

    assessment = assess_chunk(result, tone(10))

    assert assessment["flags"] == ["high_compression"]
    assert assessment["metrics"]["max_compression_ratio"] == 3.1


def test_assess_chunk_flags_low_coverage_only_with_audio() -> None:
    result = {"segments": [segment(0.0, 1.0, "はい")]}

    assert assess_chunk(result, tone(20))["flags"] == ["low_coverage"]
    assert assess_chunk(result, silence(20))["flags"] == []


def test_assess_chunk_flags_no_segments_only_with_audio() -> None:
    assert assess_chunk({"segments": []}, tone(20))["flags"] == ["no_segments"]
    assert assess_chunk({"segments": []}, silence(20))["flags"] == []


def test_quality_rank_prefers_fewer_flags_then_coverage() -> None:
    looping = {"flags": ["repeated_text"], "metrics": metrics(coverage=0.9)}
    sparse = {"flags": [], "metrics": metrics(coverage=0.4)}
    full = {"flags": [], "metrics": metrics(coverage=0.8)}

    assert min([looping, sparse, full], key=quality_rank) is full


def metrics(coverage: float) -> dict:
    return {"coverage": coverage, "high_compression_segments": 0, "max_repeat_run": 1}


def test_retry_temperatures_step_up_and_cap() -> None:
    assert retry_temperatures(1) == (0.2, 0.4)
    assert retry_temperatures(2) == (0.4, 0.6)
    assert retry_temperatures(5) == (1.0,)
