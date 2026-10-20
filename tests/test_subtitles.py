import json
from pathlib import Path

import pytest

from mlxscribe import cli
from mlxscribe.exporters.srt import write_srt
from mlxscribe.subtitles import (
    SubtitleOptions,
    clean_subtitles,
    is_cjk_text,
    wrap_text,
)

OPTIONS = SubtitleOptions()


def clean(segments: list[dict], **overrides) -> tuple[list[dict], dict]:
    options = SubtitleOptions(**overrides)
    result, summary = clean_subtitles({"segments": segments}, options)
    return result["segments"], summary


def seg(start: float, end: float, text: str, **extra) -> dict:
    return {"start": start, "end": end, "text": text, **extra}


def assert_valid(segments: list[dict], options: SubtitleOptions = OPTIONS) -> None:
    for index, segment in enumerate(segments):
        cap = options.max_line_chars_cjk if is_cjk_text(segment["text"]) else options.max_line_chars
        assert len(segment["lines"]) <= options.max_lines
        assert all(len(line) <= cap for line in segment["lines"]), segment
        assert segment["end"] - segment["start"] >= options.min_duration_seconds - 1e-9
        if index + 1 < len(segments):
            assert segment["end"] <= segments[index + 1]["start"]


def test_is_cjk_text_uses_majority_script() -> None:
    assert is_cjk_text("今日はいい天気")
    assert is_cjk_text("IT企業に勤めている")
    assert not is_cjk_text("Hello, world")
    assert not is_cjk_text("")


def test_wrap_latin_breaks_at_spaces_and_balances_lines() -> None:
    lines = wrap_text(
        "This is a fairly long English sentence that should wrap nicely, right here.",
        42,
        2,
    )
    assert lines == ["This is a fairly long English sentence", "that should wrap nicely, right here."]


def test_wrap_prefers_punctuation_over_the_exact_middle() -> None:
    assert wrap_text("今日はいい天気ですね、散歩に行きませんか", 16, 2) == [
        "今日はいい天気ですね、",
        "散歩に行きませんか",
    ]


def test_wrap_breaks_cjk_at_utterance_space() -> None:
    assert wrap_text("今日は味くらい作ってみたんだ くそ はい", 16, 2) == [
        "今日は味くらい作ってみたんだ",
        "くそ はい",
    ]


def test_wrap_avoids_splitting_katakana_words() -> None:
    lines = wrap_text("またボーナスカットだったよそっか", 10, 2)
    assert lines is not None
    assert "ボーナスカット" in lines[0] or "ボーナスカット" in lines[1]


def test_wrap_keeps_honorific_prefix_with_its_word() -> None:
    assert wrap_text("そうなんだよね家あるしさお金も自由に使えるじゃん", 16, 2) == [
        "そうなんだよね家あるしさ",
        "お金も自由に使えるじゃん",
    ]


def test_wrap_never_starts_a_line_with_closing_punctuation() -> None:
    lines = wrap_text("あいうえおかきくけこ。さしすせそ", 10, 2)
    assert lines is not None
    assert not lines[1].startswith("。")


def test_wrap_returns_none_when_text_cannot_fit() -> None:
    assert wrap_text("あ" * 40, 16, 2) is None


def test_short_segment_is_kept_as_one_line() -> None:
    segments, summary = clean([seg(0.0, 2.0, " こんにちは ")])
    assert segments == [{"id": 0, "start": 0.0, "end": 2.0, "text": "こんにちは", "lines": ["こんにちは"]}]
    assert summary["split_segments"] == 0
    assert summary["merged_segments"] == 0


def test_long_text_is_split_into_proportionally_timed_pieces() -> None:
    text = "最近帰り遅かったでしょ、少しでも稼ぎ増やすために残業頑張ってくれてるんだなって思ってたんだ。だから私も働きたいの"
    segments, summary = clean([seg(10.0, 16.0, text)])
    assert len(segments) >= 2
    assert summary["split_segments"] == 1
    assert "".join(segment["text"] for segment in segments) == text
    assert segments[0]["start"] == 10.0
    assert segments[-1]["end"] == 16.0
    # Proportional timing: the boundary sits where the text was cut.
    cut = len(segments[0]["text"]) / len(text)
    assert segments[0]["end"] == pytest.approx(10.0 + 6.0 * cut, abs=0.01)
    assert_valid(segments)


def test_split_uses_word_timestamps_when_present() -> None:
    words = [
        {"word": " This", "start": 0.0, "end": 0.4},
        {"word": " is", "start": 0.4, "end": 0.6},
        {"word": " the", "start": 0.6, "end": 0.8},
        {"word": " first", "start": 0.8, "end": 1.3},
        {"word": " sentence", "start": 1.3, "end": 2.0},
        {"word": " of", "start": 2.0, "end": 2.2},
        {"word": " many.", "start": 2.2, "end": 3.0},
        {"word": " And", "start": 9.0, "end": 9.3},
        {"word": " here", "start": 9.3, "end": 9.6},
        {"word": " comes", "start": 9.6, "end": 10.0},
        {"word": " the", "start": 10.0, "end": 10.2},
        {"word": " second", "start": 10.2, "end": 10.7},
        {"word": " one.", "start": 10.7, "end": 11.5},
    ]
    text = "".join(word["word"] for word in words)
    segments, summary = clean([seg(0.0, 12.0, text, words=words)], max_line_chars=20)
    assert summary["split_with_word_timestamps"] == 1
    assert [segment["text"] for segment in segments] == [
        "This is the first sentence of many.",
        "And here comes the second one.",
    ]
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 3.0
    assert segments[1]["start"] == 9.0
    assert segments[1]["end"] == 11.5


def test_over_long_duration_is_split_at_sentence_breaks() -> None:
    text = "それでは始めましょう。まずは自己紹介からお願いします。"
    segments, summary = clean([seg(0.0, 11.0, text)])
    assert [segment["text"] for segment in segments] == [
        "それでは始めましょう。",
        "まずは自己紹介からお願いします。",
    ]
    assert all(segment["end"] - segment["start"] <= OPTIONS.max_duration_seconds for segment in segments)
    assert summary["remaining_long_cues"] == 0


def test_over_long_duration_without_a_pause_in_the_text_is_not_split() -> None:
    segments, summary = clean([seg(0.0, 20.0, "まずは自己紹介からお願いします")])
    assert len(segments) == 1
    assert summary["remaining_long_cues"] == 1


def test_over_long_duration_with_little_text_is_left_alone() -> None:
    segments, summary = clean([seg(0.0, 19.0, "おかえりなさい")])
    assert [segment["text"] for segment in segments] == ["おかえりなさい"]
    assert summary["remaining_long_cues"] == 1


def test_over_long_duration_is_trimmed_to_word_timestamps() -> None:
    words = [{"word": "おかえり", "start": 17.0, "end": 17.6}, {"word": "なさい", "start": 17.6, "end": 18.4}]
    segments, summary = clean([seg(0.0, 19.0, "おかえりなさい", words=words)])
    assert (segments[0]["start"], segments[0]["end"]) == (17.0, 18.4)
    assert summary["trimmed_to_words"] == 1
    assert summary["remaining_long_cues"] == 0


def test_short_adjacent_segments_are_merged() -> None:
    segments, summary = clean(
        [seg(0.0, 1.5, "ただいま"), seg(1.6, 2.2, "はい"), seg(5.0, 7.0, "ご飯もできてるからさ")]
    )
    assert [segment["text"] for segment in segments] == ["ただいま はい", "ご飯もできてるからさ"]
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 2.2
    assert summary["merged_segments"] == 1


def test_merge_joins_cjk_after_punctuation_without_space() -> None:
    segments, _ = clean([seg(0.0, 0.5, "うん、"), seg(0.6, 2.0, "美味しいよ")])
    assert segments[0]["text"] == "うん、美味しいよ"


def test_short_segments_are_not_merged_across_a_large_gap() -> None:
    segments, summary = clean([seg(0.0, 1.5, "はい"), seg(3.0, 4.5, "いいえ")])
    assert len(segments) == 2
    assert summary["merged_segments"] == 0


def test_short_segments_are_not_merged_when_result_would_not_fit() -> None:
    long_text = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
    segments, _ = clean([seg(0.0, 4.0, long_text), seg(4.1, 4.5, "はい")])
    assert [segment["text"] for segment in segments][-1] == "はい"
    assert_valid(segments)


def test_min_duration_extends_into_the_following_gap() -> None:
    segments, summary = clean(
        [seg(0.0, 0.3, "Good morning everyone"), seg(3.0, 5.0, "Welcome to the show")],
        merge_gap_seconds=0.0,
    )
    assert segments[0]["end"] == pytest.approx(1.0)
    assert summary["extended_cues"] == 1
    assert_valid(segments)


def test_min_duration_never_overlaps_the_next_cue() -> None:
    segments, _ = clean(
        [
            seg(0.0, 2.0, "The first sentence is here."),
            seg(2.0, 2.3, "A quick second sentence follows."),
            seg(2.4, 4.0, "Then a third sentence closes it."),
        ],
        merge_gap_seconds=0.0,
        max_line_chars=40,
        max_lines=1,
    )
    assert len(segments) == 3
    assert_valid(segments, SubtitleOptions(max_line_chars=40, max_lines=1))
    assert segments[1]["start"] >= segments[0]["end"]


def test_overlapping_input_segments_are_fixed() -> None:
    segments, summary = clean([seg(0.0, 3.0, "First line here"), seg(2.0, 4.0, "Second line here")])
    assert summary["overlaps_fixed"] == 1
    assert segments[0]["end"] <= segments[1]["start"]


def test_empty_segments_are_dropped() -> None:
    segments, summary = clean([seg(0.0, 1.0, "  "), seg(1.0, 3.0, "Hello there")])
    assert [segment["text"] for segment in segments] == ["Hello there"]
    assert summary["dropped_empty"] == 1


def test_disabled_cleanup_returns_result_unchanged() -> None:
    result = {"segments": [seg(0.0, 0.1, "x")]}
    cleaned, summary = clean_subtitles(result, SubtitleOptions(enabled=False))
    assert cleaned is result
    assert summary == {"enabled": False}


def test_dense_dialogue_meets_all_limits() -> None:
    lines = [
        "はい",
        "うん",
        "そうなんだ",
        "えっ本当に？それは知らなかったな、教えてくれてありがとう",
        "いいえ",
        "じゃあまた明日学校で会おうね、遅れないでよ",
        "うん",
    ]
    segments = [seg(index * 0.6, index * 0.6 + 0.5, text) for index, text in enumerate(lines)]
    cleaned, summary = clean(segments)
    assert_valid(cleaned)
    assert summary["remaining_short_cues"] == 0
    assert summary["remaining_overlaps"] == 0


def test_write_srt_uses_wrapped_lines(tmp_path: Path) -> None:
    output = tmp_path / "sample.srt"
    write_srt(output, {"segments": [{"start": 0.0, "end": 2.0, "text": "a b", "lines": ["a", "b"]}]})
    assert output.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:02,000\na\nb\n"


def test_filter_command_cleans_srt_and_txt_but_keeps_raw_json(tmp_path: Path) -> None:
    raw = {
        "language": "ja",
        "text": "",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "ただいま"},
            {"id": 1, "start": 1.6, "end": 2.2, "text": "はい"},
            {"id": 2, "start": 5.0, "end": 9.0, "text": "最近帰り遅かったでしょ、少しでも稼ぎ増やすために残業頑張ってくれてるんだなって"},
        ],
    }
    source = tmp_path / "movie.ja.json"
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out"

    assert cli.main(["filter", str(source), "--output", str(output)]) == 0

    written = json.loads((output / "movie.ja.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in written["segments"]] == [
        segment["text"] for segment in raw["segments"]
    ]
    assert written["subtitles"]["merged_segments"] == 1
    assert written["subtitles"]["split_segments"] == 1

    srt = (output / "movie.ja.srt").read_text(encoding="utf-8")
    assert "ただいま はい" in srt
    text_lines = [line for line in srt.splitlines() if line and "-->" not in line and not line.isdigit()]
    assert all(len(line) <= 16 for line in text_lines)
    txt = (output / "movie.ja.txt").read_text(encoding="utf-8")
    assert txt.splitlines()[0] == "[00:00:00.000] ただいま はい"


def test_filter_command_can_disable_cleanup(tmp_path: Path) -> None:
    raw = {"language": "en", "segments": [{"id": 0, "start": 0.0, "end": 0.2, "text": "Hi"}]}
    source = tmp_path / "movie.en.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "out"

    assert cli.main(["filter", str(source), "--output", str(output), "--no-subtitle-cleanup"]) == 0

    assert (output / "movie.en.srt").read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:00,200\nHi\n"


def test_subtitle_options_come_from_cli_flags() -> None:
    args = cli.build_parser().parse_args(
        [
            "filter",
            "in.json",
            "--output",
            "out",
            "--max-lines",
            "1",
            "--max-line-chars",
            "30",
            "--max-line-chars-cjk",
            "12",
            "--min-cue-seconds",
            "0.8",
            "--max-cue-seconds",
            "6",
            "--merge-gap-seconds",
            "0.3",
        ]
    )
    assert cli.subtitle_options_from_args(args) == SubtitleOptions(
        max_lines=1,
        max_line_chars=30,
        max_line_chars_cjk=12,
        min_duration_seconds=0.8,
        max_duration_seconds=6.0,
        merge_gap_seconds=0.3,
    )


def test_max_lines_rejects_zero(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["filter", "in.json", "--output", "out", "--max-lines", "0"])


def test_transcribe_records_subtitle_summary_in_benchmark(tmp_path, monkeypatch) -> None:
    import numpy as np

    input_path = tmp_path / "movie.mp4"
    input_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    raw_segments = [
        {"id": 0, "start": 0.0, "end": 1.2, "text": "ただいま"},
        {"id": 1, "start": 1.3, "end": 1.6, "text": "はい"},
    ]

    monkeypatch.setattr(cli, "probe_duration", lambda path: 10.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
    monkeypatch.setattr(cli, "load_wav", lambda path: np.zeros(10 * 16000, dtype=np.float32))
    monkeypatch.setattr(
        cli,
        "transcribe_audio",
        lambda audio, **kwargs: {"language": "ja", "segments": [dict(s) for s in raw_segments]},
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    assert cli.main(["transcribe", str(input_path), "--output", str(output_dir)]) == 0

    benchmark = json.loads((output_dir / "movie.benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["subtitles"]["enabled"] is True
    assert benchmark["subtitles"]["merged_segments"] == 1
    transcript = json.loads((output_dir / "movie.ja.json").read_text(encoding="utf-8"))
    assert len(transcript["segments"]) == 2
    assert "ただいま はい" in (output_dir / "movie.ja.srt").read_text(encoding="utf-8")
