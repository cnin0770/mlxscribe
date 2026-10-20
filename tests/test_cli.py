import json

import numpy as np
import pytest

from mlxscribe import cli
from mlxscribe.cli import (
    LONG_FORM_THRESHOLD_SECONDS,
    build_parser,
    normalize_language,
    resolve_context_mode,
)


def test_normalize_language_maps_auto_to_none() -> None:
    assert normalize_language("auto") is None
    assert normalize_language("AUTO") is None


def test_normalize_language_preserves_explicit_language() -> None:
    assert normalize_language("ja") == "ja"


def test_resolve_context_auto_keeps_context_for_short_inputs() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=LONG_FORM_THRESHOLD_SECONDS,
    )

    assert decision["enabled"] is True
    assert decision["requested"] == "auto"


def test_resolve_context_auto_disables_context_for_long_inputs() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=LONG_FORM_THRESHOLD_SECONDS + 1,
    )

    assert decision["enabled"] is False
    assert decision["requested"] == "auto"


def test_resolve_context_explicit_modes_override_duration() -> None:
    assert resolve_context_mode(requested_mode="on", duration_seconds=999999)["enabled"] is True
    assert resolve_context_mode(requested_mode="off", duration_seconds=1)["enabled"] is False


def test_resolve_context_legacy_override_wins() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=1,
        explicit_override=False,
    )

    assert decision["enabled"] is False
    assert decision["reason"] == "explicit legacy override"


def test_chunk_minutes_rejects_non_positive_values(capsys) -> None:
    for value in ("0", "-5", "abc"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["transcribe", "movie.mp4", "--output", "out", "--chunk-minutes", value]
            )
        assert "argument --chunk-minutes" in capsys.readouterr().err


def test_transcribe_with_chunks_merges_results(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "movie.mp4"
    input_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    calls = []

    def fake_transcribe_audio(audio, **kwargs):
        calls.append(len(audio) / 16000)
        return {
            "language": "ja",
            "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": f"chunk {len(calls)}"}],
        }

    monkeypatch.setattr(cli, "probe_duration", lambda path: 150.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
    monkeypatch.setattr(cli, "load_wav", lambda path: np.zeros(150 * 16000, dtype=np.float32))
    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    exit_code = cli.main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(output_dir),
            "--formats",
            "json",
            "--chunk-minutes",
            "1",
        ]
    )

    assert exit_code == 0
    # Silent audio: every point ties, so boundaries stay at the planned positions.
    assert calls == pytest.approx([60.0, 60.0, 30.0], abs=0.1)

    benchmark = json.loads((output_dir / "movie.benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["chunking"]["enabled"] is True
    assert benchmark["chunking"]["chunk_minutes"] == 1.0
    assert [chunk["segments"] for chunk in benchmark["chunking"]["chunks"]] == [1, 1, 1]
    assert all("elapsed_seconds" in chunk for chunk in benchmark["chunking"]["chunks"])

    transcript = json.loads((output_dir / "movie.json").read_text(encoding="utf-8"))
    starts = [segment["start"] for segment in transcript["segments"]]
    assert starts == pytest.approx([1.0, 61.0, 121.0], abs=0.1)


LOOP = [{"id": index, "start": index, "end": index + 1, "text": "ループ"} for index in range(8)]
CLEAN = [{"id": 0, "start": 0.0, "end": 50.0, "text": "きれいな文章"}]


def run_one_chunk(tmp_path, monkeypatch, responses: list[list[dict]], *extra: str) -> tuple:
    input_path = tmp_path / "movie.mp4"
    input_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    temperatures = []

    def fake_transcribe_audio(audio, **kwargs):
        temperatures.append(kwargs["temperature"])
        return {"language": "ja", "segments": responses[len(temperatures) - 1]}

    monkeypatch.setattr(cli, "probe_duration", lambda path: 60.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
    monkeypatch.setattr(cli, "load_wav", lambda path: np.zeros(60 * 16000, dtype=np.float32))
    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    exit_code = cli.main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(output_dir),
            "--formats",
            "json",
            "--chunk-minutes",
            "1",
            *extra,
        ]
    )
    assert exit_code == 0
    benchmark = json.loads((output_dir / "movie.benchmark.json").read_text(encoding="utf-8"))
    transcript = json.loads((output_dir / "movie.json").read_text(encoding="utf-8"))
    return temperatures, benchmark, transcript


def test_chunked_retry_keeps_better_attempt(tmp_path, monkeypatch) -> None:
    temperatures, benchmark, transcript = run_one_chunk(tmp_path, monkeypatch, [LOOP, CLEAN])

    assert temperatures == [0.0, (0.2, 0.4)]
    quality = benchmark["chunking"]["chunks"][0]["quality"]
    assert quality["flagged"] is True
    assert quality["chosen_attempt"] == 1
    assert quality["flags"] == []
    assert [attempt["flags"] for attempt in quality["attempts"]] == [["repeated_text"], []]
    assert quality["attempts"][1]["temperature"] == [0.2, 0.4]
    assert benchmark["chunking"]["flagged_chunks"] == 1
    assert benchmark["chunking"]["retried_chunks"] == 1
    assert [segment["text"] for segment in transcript["segments"]] == ["きれいな文章"]


def test_chunked_retry_keeps_original_when_retry_is_no_better(tmp_path, monkeypatch) -> None:
    temperatures, benchmark, transcript = run_one_chunk(tmp_path, monkeypatch, [LOOP, LOOP])

    assert len(temperatures) == 2
    quality = benchmark["chunking"]["chunks"][0]["quality"]
    assert quality["chosen_attempt"] == 0
    assert quality["flags"] == ["repeated_text"]
    assert len(transcript["segments"]) == len(LOOP)


def test_chunked_retry_skips_clean_chunks(tmp_path, monkeypatch) -> None:
    temperatures, benchmark, _ = run_one_chunk(tmp_path, monkeypatch, [CLEAN])

    assert temperatures == [0.0]
    quality = benchmark["chunking"]["chunks"][0]["quality"]
    assert quality["flagged"] is False
    assert len(quality["attempts"]) == 1
    assert benchmark["chunking"]["retried_chunks"] == 0


def test_chunk_retries_zero_disables_retry(tmp_path, monkeypatch) -> None:
    temperatures, benchmark, _ = run_one_chunk(
        tmp_path, monkeypatch, [LOOP], "--chunk-retries", "0"
    )

    assert temperatures == [0.0]
    assert benchmark["chunking"]["chunk_retries"] == 0
    assert benchmark["chunking"]["flagged_chunks"] == 1
    assert benchmark["chunking"]["retried_chunks"] == 0


def test_chunk_retries_rejects_negative_values(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["transcribe", "movie.mp4", "--chunk-retries", "-1"])
    assert "--chunk-retries" in capsys.readouterr().err


def test_resolve_chunk_mode_auto_chunks_only_long_inputs() -> None:
    long_input = cli.resolve_chunk_mode(
        requested="auto", duration_seconds=LONG_FORM_THRESHOLD_SECONDS + 1
    )
    short_input = cli.resolve_chunk_mode(
        requested="auto", duration_seconds=LONG_FORM_THRESHOLD_SECONDS
    )

    assert long_input["minutes"] == cli.DEFAULT_CHUNK_MINUTES
    assert short_input["minutes"] is None


def test_resolve_chunk_mode_explicit_values_override_duration() -> None:
    assert cli.resolve_chunk_mode(requested="off", duration_seconds=99999)["minutes"] is None
    assert cli.resolve_chunk_mode(requested=2.0, duration_seconds=60)["minutes"] == 2.0
    assert cli.resolve_chunk_mode(requested="auto", duration_seconds=None)["minutes"] is None


def test_chunk_minutes_accepts_auto_off_and_numbers() -> None:
    parse = lambda *extra: build_parser().parse_args(["transcribe", "movie.mp4", "--output", "out", *extra])

    assert parse().chunk_minutes == "auto"
    assert parse("--chunk-minutes", "OFF").chunk_minutes == "off"
    assert parse("--chunk-minutes", "2.5").chunk_minutes == 2.5


def test_temperature_accepts_single_value_or_fallback_list() -> None:
    parse = lambda *extra: build_parser().parse_args(["transcribe", "movie.mp4", "--output", "out", *extra])

    assert parse().temperature == 0.0
    assert parse("--temperature", "0.2").temperature == 0.2
    assert parse("--temperature", "0,0.2,0.4").temperature == (0.0, 0.2, 0.4)
    with pytest.raises(SystemExit):
        parse("--temperature", "0,1.5")
