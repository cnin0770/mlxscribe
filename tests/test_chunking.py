from mlxscribe.chunking import Chunk, format_clip_timestamps, merge_chunk_results, plan_chunks


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


def test_format_clip_timestamps() -> None:
    assert format_clip_timestamps(Chunk(index=0, start=12.3456, end=78.9)) == "12.346,78.900"


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
