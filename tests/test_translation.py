from pathlib import Path

from mlxscribe.translation import (
    estimate_translation_tokens,
    parse_translation_response,
    write_translation_outputs,
)


def test_estimate_translation_tokens_counts_segments() -> None:
    estimate = estimate_translation_tokens(
        [
            {"id": 1, "text": "こんにちは"},
            {"id": 2, "text": "よろしくお願いします"},
        ],
        target_language="en",
    )

    assert estimate.source_text_tokens > 0
    assert estimate.prompt_tokens > estimate.source_text_tokens
    assert estimate.expected_output_tokens > 0
    assert estimate.total_tokens == estimate.prompt_tokens + estimate.expected_output_tokens


def test_parse_translation_response_accepts_json_array() -> None:
    parsed = parse_translation_response('[{"id": 1, "text": "Hello"}]')

    assert parsed == [{"id": 1, "text": "Hello"}]


def test_write_translation_outputs_keeps_language_in_names(tmp_path) -> None:
    result = {"language": "en", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}]}
    paths = write_translation_outputs(result=result, output_dir=tmp_path, stem="movie", target_language="en")

    assert {key: Path(value).name for key, value in paths.items()} == {
        "json": "movie.en.json",
        "srt": "movie.en.srt",
        "txt": "movie.en.txt",
    }
