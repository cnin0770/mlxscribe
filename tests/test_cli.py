import json

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
            build_parser().parse_args(["transcribe", "movie.mp4", "--chunk-minutes", value])
    assert "--chunk-minutes" in capsys.readouterr().err


def test_transcribe_with_chunks_merges_results(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "movie.mp4"
    input_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    calls = []

    def fake_transcribe_audio(audio_path, **kwargs):
        calls.append(kwargs["clip_timestamps"])
        start = float(kwargs["clip_timestamps"].split(",")[0])
        return {
            "language": "ja",
            "segments": [{"id": 0, "start": start + 1, "end": start + 2, "text": f"at {start:g}"}],
        }

    monkeypatch.setattr(cli, "probe_duration", lambda path: 150.0)
    monkeypatch.setattr(cli, "extract_audio", lambda source, target: None)
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
    assert calls == ["0.000,60.000", "60.000,120.000", "120.000,150.000"]

    benchmark = json.loads((output_dir / "movie.benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["chunking"]["enabled"] is True
    assert benchmark["chunking"]["chunk_minutes"] == 1.0
    assert [chunk["segments"] for chunk in benchmark["chunking"]["chunks"]] == [1, 1, 1]
    assert all("elapsed_seconds" in chunk for chunk in benchmark["chunking"]["chunks"])

    transcript = json.loads((output_dir / "movie.json").read_text(encoding="utf-8"))
    assert [segment["start"] for segment in transcript["segments"]] == [1.0, 61.0, 121.0]
