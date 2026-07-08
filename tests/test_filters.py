from mlxscribe.filters import FilterOptions, filter_transcript, is_repetitive_text


def test_filter_drops_leading_repetition() -> None:
    result = {
        "text": "ハラハラハラハラこんにちは",
        "language": "ja",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 10.0,
                "text": "ハラハラハラハラ",
                "avg_logprob": -0.4,
                "no_speech_prob": 0.0,
            },
            {
                "id": 1,
                "start": 10.0,
                "end": 12.0,
                "text": "こんにちは",
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            },
        ],
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped_segments"] == 1
    assert summary["dropped"][0]["reason"] == "leading_repetition"
    assert filtered["segments"] == [
        {
            "id": 0,
            "start": 10.0,
            "end": 12.0,
            "text": "こんにちは",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.0,
        }
    ]
    assert filtered["text"] == "こんにちは"


def test_filter_keeps_repetition_after_leading_window() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 20.0,
                "end": 24.0,
                "text": "ハラハラハラハラ",
                "avg_logprob": -0.4,
                "no_speech_prob": 0.0,
            }
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped_segments"] == 0
    assert len(filtered["segments"]) == 1


def test_filter_drops_low_confidence_silence() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 30.0,
                "end": 31.0,
                "text": "noise",
                "avg_logprob": -1.5,
                "no_speech_prob": 0.95,
            }
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped"][0]["reason"] == "low_confidence_silence"
    assert filtered["segments"] == []


def test_filter_drops_high_compression_repetition_after_leading_window() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 90.0,
                "end": 120.0,
                "text": "あっ、あっ、あっ、あっ、あっ、あっ",
                "compression_ratio": 20.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped"][0]["reason"] == "high_compression_repetition"
    assert filtered["segments"] == []


def test_filter_drops_high_compression_repeated_phrase_with_prefix() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 75.0,
                "end": 105.0,
                "text": "そういう時は、ちょっと、ちょっと、ちょっと、ちょっと、ちょっと",
                "compression_ratio": 20.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped"][0]["reason"] == "high_compression_repetition"
    assert filtered["segments"] == []


def test_filter_keeps_high_compression_non_repetitive_text() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 90.0,
                "end": 93.0,
                "text": "おしゃべりは全然、好きです。はい。",
                "compression_ratio": 5.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped_segments"] == 0
    assert len(filtered["segments"]) == 1


def test_filter_drops_high_compression_adjacent_duplicate_run() -> None:
    result = {
        "segments": [
            {
                "id": 0,
                "start": 115.0,
                "end": 116.0,
                "text": "お茶、似合うでしょ。",
                "compression_ratio": 11.0,
            },
            {
                "id": 1,
                "start": 116.0,
                "end": 117.0,
                "text": "お茶、似合うでしょ。",
                "compression_ratio": 11.0,
            },
            {
                "id": 2,
                "start": 117.0,
                "end": 118.0,
                "text": "お茶、似合うでしょ。",
                "compression_ratio": 11.0,
            },
        ]
    }

    filtered, summary = filter_transcript(result, FilterOptions(enabled=True))

    assert summary["dropped"][0]["reason"] == "repeated_adjacent_text"
    assert [segment["id"] for segment in filtered["segments"]] == [0, 1]


def test_repetitive_text_detects_repeated_ngram() -> None:
    assert is_repetitive_text("ハラハラハラハラ", FilterOptions())
    assert is_repetitive_text("あっ、あっ、あっ、あっ", FilterOptions())
    assert is_repetitive_text("そういう時は、ちょっと、ちょっと、ちょっと、ちょっと", FilterOptions())
    assert is_repetitive_text("お前のお前のお前のお前のお前のお前のお", FilterOptions())
    assert is_repetitive_text("キンチンキンキンキンキンキンキンキン", FilterOptions())
    assert is_repetitive_text("はぁはぁはぁはぁはぁはぁはぁ", FilterOptions())
    assert not is_repetitive_text("こんにちは", FilterOptions())
