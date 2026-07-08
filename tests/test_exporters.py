from pathlib import Path

from mlxscribe.exporters.srt import format_srt_timestamp, write_srt
from mlxscribe.exporters.txt import write_txt
from mlxscribe.formats import parse_formats


def test_format_srt_timestamp_rounds_to_milliseconds() -> None:
    assert format_srt_timestamp(3723.4567) == "01:02:03,457"


def test_write_srt(tmp_path: Path) -> None:
    output = tmp_path / "sample.srt"
    write_srt(
        output,
        {
            "segments": [
                {"start": 0.0, "end": 1.25, "text": "こんにちは"},
                {"start": 1.5, "end": 3.0, "text": "世界"},
            ]
        },
    )

    assert output.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "こんにちは\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "世界\n"
    )


def test_parse_formats_deduplicates_and_preserves_order() -> None:
    assert parse_formats("srt,json,srt,txt") == ("srt", "json", "txt")


def test_write_txt_prefers_timestamped_segments(tmp_path: Path) -> None:
    output = tmp_path / "sample.txt"
    write_txt(
        output,
        {
            "text": "ignored dense text",
            "segments": [
                {"start": 0.0, "end": 1.25, "text": "こんにちは"},
                {"start": 1.5, "end": 3.0, "text": "世界"},
            ],
        },
    )

    assert output.read_text(encoding="utf-8") == (
        "[00:00:00.000] こんにちは\n"
        "[00:00:01.500] 世界\n"
    )
