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
from mlxscribe.transcripts import transcript_stem


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

    transcript = json.loads((output_dir / "movie.ja.json").read_text(encoding="utf-8"))
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
    transcript = json.loads((output_dir / "movie.ja.json").read_text(encoding="utf-8"))
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


def test_filter_keeps_language_in_output_names(tmp_path) -> None:
    source = tmp_path / "movie.ja.json"
    source.write_text(
        json.dumps({"language": "ja", "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "こんにちは"}]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "filtered"

    assert cli.main(["filter", str(source), "--output", str(output_dir)]) == 0
    assert sorted(path.name for path in output_dir.iterdir()) == ["movie.ja.json", "movie.ja.srt", "movie.ja.txt"]


def test_filter_refuses_to_overwrite_its_input(tmp_path) -> None:
    source = tmp_path / "movie.ja.json"
    original = json.dumps({"language": "ja", "segments": []})
    source.write_text(original, encoding="utf-8")

    assert cli.main(["filter", str(source), "--output", str(tmp_path)]) == 1
    assert source.read_text(encoding="utf-8") == original


def test_transcript_stem_round_trips_output_name(tmp_path) -> None:
    assert transcript_stem(tmp_path / "movie.ja.json", "ja") == "movie"


def run_resumable(tmp_path, monkeypatch, *extra: str, fail_at: int | None = None) -> tuple[int, list[int]]:
    """Transcribe a 3-minute file in 1-minute chunks; raise KeyboardInterrupt on chunk ``fail_at``."""
    input_path = tmp_path / "movie.mp4"
    if not input_path.exists():
        input_path.write_bytes(b"movie")
    calls = []

    def fake_transcribe_audio(audio, **kwargs):
        if len(calls) == fail_at:
            raise KeyboardInterrupt
        calls.append(len(calls))
        return {
            "language": "ja",
            "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": f"{kwargs['model']} {len(calls)}"}],
        }

    monkeypatch.setattr(cli, "probe_duration", lambda path: 180.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
    monkeypatch.setattr(cli, "load_wav", lambda path: np.zeros(180 * 16000, dtype=np.float32))
    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    exit_code = cli.main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(tmp_path / "out"),
            "--formats",
            "json",
            "--chunk-minutes",
            "1",
            *extra,
        ]
    )
    return exit_code, calls


def read_texts(tmp_path) -> list[str]:
    transcript = json.loads((tmp_path / "out" / "movie.ja.json").read_text(encoding="utf-8"))
    return [segment["text"] for segment in transcript["segments"]]


def test_interrupted_transcription_resumes_finished_chunks(tmp_path, monkeypatch, capsys) -> None:
    work = tmp_path / "out" / ".mlxscribe-work" / "movie" / "transcribe"
    progress = tmp_path / "progress.jsonl"

    exit_code, calls = run_resumable(tmp_path, monkeypatch, fail_at=2)
    assert exit_code == 130
    assert calls == [0, 1]
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "transcribe"
    assert manifest["total"] == 3
    assert manifest["completed"] == ["chunk-0000", "chunk-0001"]

    exit_code, calls = run_resumable(tmp_path, monkeypatch, "--progress-jsonl", str(progress))
    assert exit_code == 0
    assert calls == [0]  # only the third chunk is transcribed again
    assert "Resuming from" in capsys.readouterr().out
    assert read_texts(tmp_path) == [f"{cli.DEFAULT_MODEL} {n}" for n in (1, 2, 1)]

    benchmark = json.loads((tmp_path / "out" / "movie.benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["chunking"]["resumed_chunks"] == 2
    assert [chunk["resumed"] for chunk in benchmark["chunking"]["chunks"]] == [True, True, False]
    assert not (tmp_path / "out" / ".mlxscribe-work").exists()

    events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    chunk_events = [(event["index"], event["status"]) for event in events if event["event"] == "chunk"]
    assert chunk_events == [(0, "resumed"), (1, "resumed"), (2, "started"), (2, "done")]
    assert events[-1]["event"] == "job" and events[-1]["status"] == "complete"


def test_changed_setting_invalidates_transcription_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    exit_code, _ = run_resumable(tmp_path, monkeypatch, fail_at=2)
    assert exit_code == 130
    capsys.readouterr()

    exit_code, calls = run_resumable(tmp_path, monkeypatch, "--model", "other-model")
    assert exit_code == 0
    assert calls == [0, 1, 2]
    assert "different settings (model); starting fresh" in capsys.readouterr().out
    assert read_texts(tmp_path) == [f"other-model {n}" for n in (1, 2, 3)]


def test_changed_input_invalidates_transcription_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    run_resumable(tmp_path, monkeypatch, fail_at=2)
    (tmp_path / "movie.mp4").write_bytes(b"a different movie")
    capsys.readouterr()

    _, calls = run_resumable(tmp_path, monkeypatch)
    assert calls == [0, 1, 2]
    assert "different settings (input)" in capsys.readouterr().out


def test_no_resume_discards_checkpoint_and_keep_work_keeps_it(tmp_path, monkeypatch) -> None:
    run_resumable(tmp_path, monkeypatch, fail_at=2)

    exit_code, calls = run_resumable(tmp_path, monkeypatch, "--no-resume", "--keep-work")
    assert exit_code == 0
    assert calls == [0, 1, 2]
    manifest_path = tmp_path / "out" / ".mlxscribe-work" / "movie" / "transcribe" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["completed"]) == 3


def test_work_dir_override_is_used_and_not_removed(tmp_path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    run_resumable(tmp_path, monkeypatch, "--work-dir", str(work_dir), fail_at=1)
    assert (work_dir / "transcribe" / "chunk-0000.json").exists()

    _, calls = run_resumable(tmp_path, monkeypatch, "--work-dir", str(work_dir))
    assert calls == [0, 1]
    assert work_dir.exists() and not (work_dir / "transcribe").exists()


def capture_engine_kwargs(tmp_path, monkeypatch, *extra: str) -> list[dict]:
    calls = []

    def fake_transcribe_audio(audio, **kwargs):
        calls.append(kwargs)
        return {"language": "ja", "segments": CLEAN}

    input_path = tmp_path / "movie.mp4"
    input_path.write_bytes(b"")
    monkeypatch.setattr(cli, "probe_duration", lambda path: 60.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
    monkeypatch.setattr(cli, "load_wav", lambda path: np.zeros(60 * 16000, dtype=np.float32))
    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    argv = ["transcribe", str(input_path), "--output", str(tmp_path / "out"), "--formats", "json"]
    assert cli.main([*argv, *extra]) == 0
    return calls


def test_skip_silence_is_on_by_default_and_turns_on_word_timestamps(tmp_path, monkeypatch) -> None:
    calls = capture_engine_kwargs(tmp_path, monkeypatch, "--chunk-minutes", "1")

    assert calls[0]["skip_silence"] == cli.DEFAULT_SKIP_SILENCE_SECONDS
    assert calls[0]["word_timestamps"] is True
    benchmark = json.loads((tmp_path / "out" / "movie.benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["skip_silence"] == cli.DEFAULT_SKIP_SILENCE_SECONDS


def test_skip_silence_off_keeps_the_fast_path(tmp_path, monkeypatch) -> None:
    calls = capture_engine_kwargs(tmp_path, monkeypatch, "--skip-silence", "off")

    assert calls[0]["skip_silence"] is None
    assert calls[0]["word_timestamps"] is False


def test_chunked_transcription_does_not_list_chunks(tmp_path, monkeypatch, capsys) -> None:
    run_one_chunk(tmp_path, monkeypatch, [LOOP, CLEAN])

    out = capsys.readouterr().out
    assert "Chunks: 1 x 1 minute(s)" in out
    assert "chunk 1/1" not in out
    assert "Re-transcribing 00:00.00-01:00.00 at temperature 0.2, 0.4: the last pass had the same line repeated" in out
