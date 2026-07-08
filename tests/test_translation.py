from mlxscribe.translation import (
    estimate_translation_tokens,
    parse_translation_response,
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
