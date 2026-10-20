import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxscribe import cli, translation
from mlxscribe.jobs import Checkpoint
from mlxscribe.translation import (
    estimate_translation_tokens,
    parse_translation_response,
    translate_segments_openai,
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


class FakeClient:
    """Stands in for openai.OpenAI; ``failures`` maps a call number to the error it raises."""

    def __init__(self, failures: dict[int, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.calls = 0
        self.batches: list[list[int]] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, *, model, instructions, input):
        self.calls += 1
        if self.calls in self.failures:
            raise self.failures[self.calls]
        items = json.loads(input)
        self.batches.append([item["id"] for item in items])
        return SimpleNamespace(
            output_text=json.dumps([{"id": item["id"], "text": f"EN {item['text']}"} for item in items]),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


SEGMENTS = [{"id": index, "start": float(index), "end": index + 1.0, "text": f"line {index}"} for index in range(5)]


def open_checkpoint(path: Path, **overrides) -> Checkpoint:
    settings = {"source_language": "ja", "target_language": "en", "model": "m", "max_segments_per_batch": 2}
    settings.update(overrides)
    checkpoint, _ = Checkpoint.open(
        path,
        kind="translate",
        fingerprint=translation.translation_fingerprint(SEGMENTS, **settings),
        total=3,
        resume=True,
    )
    return checkpoint


def translate(client: FakeClient, checkpoint: Checkpoint, **kwargs):
    return translate_segments_openai(
        SEGMENTS,
        target_language="en",
        model="m",
        max_segments_per_batch=2,
        client=client,
        checkpoint=checkpoint,
        **{"sleep": lambda seconds: None, **kwargs},
    )


def test_failed_batch_retries_with_backoff(tmp_path) -> None:
    delays = []
    client = FakeClient({2: TimeoutError("timeout"), 3: ValueError("bad gateway")})

    translated, usage = translate(client, open_checkpoint(tmp_path / "work"), sleep=delays.append, retry_delay=1.5)

    assert delays == [1.5, 3.0]
    assert client.batches == [[0, 1], [2, 3], [4]]
    assert [segment["text"] for segment in translated] == [f"EN line {index}" for index in range(5)]
    assert translated[0]["source_text"] == "line 0"
    assert usage == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45, "batches": 3, "resumed_batches": 0}


def test_failed_batch_saves_finished_batches_and_resumes(tmp_path) -> None:
    work = tmp_path / "work"
    failing = FakeClient({2: TimeoutError("t1"), 3: TimeoutError("t2"), 4: TimeoutError("t3")})
    with pytest.raises(RuntimeError, match="batch 2/3 failed after 3 attempt"):
        translate(failing, open_checkpoint(work))
    assert failing.batches == [[0, 1]]
    assert (work / "batch-0000.json").exists()

    client = FakeClient()
    translated, usage = translate(client, open_checkpoint(work))

    assert client.batches == [[2, 3], [4]]  # the paid-for first batch is not requested again
    assert [segment["text"] for segment in translated] == [f"EN line {index}" for index in range(5)]
    assert usage["batches"] == 3
    assert usage["resumed_batches"] == 1
    assert usage["total_tokens"] == 45


def test_changed_translation_setting_invalidates_saved_batches(tmp_path) -> None:
    work = tmp_path / "work"
    with pytest.raises(RuntimeError):
        translate(FakeClient({2: TimeoutError("t")}), open_checkpoint(work), max_attempts=1)

    client = FakeClient()
    _, usage = translate(client, open_checkpoint(work, model="other"))

    assert client.batches == [[0, 1], [2, 3], [4]]
    assert usage["resumed_batches"] == 0


def test_translate_command_resumes_after_failure_and_cleans_up(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "movie.ja.json"
    source.write_text(json.dumps({"language": "ja", "segments": SEGMENTS}), encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    command = [
        "translate", str(source), "--to", "en", "--engine", "openai", "--output", str(output_dir),
        "--batch-size", "2", "--max-attempts", "2", "--retry-delay", "0",
    ]

    failing = FakeClient({2: TimeoutError("t1"), 3: TimeoutError("t2")})
    monkeypatch.setattr(translation, "create_openai_client", lambda: failing)
    assert cli.main(command) == 1
    assert "rerun the same command to resume" in capsys.readouterr().err
    assert (output_dir / ".mlxscribe-work" / "movie" / "translate.en" / "batch-0000.json").exists()

    client = FakeClient()
    monkeypatch.setattr(translation, "create_openai_client", lambda: client)
    assert cli.main(command) == 0
    out = capsys.readouterr().out
    assert "Resuming from" in out
    assert "Resumed batches: 1/3" in out
    assert client.batches == [[2, 3], [4]]
    result = json.loads((output_dir / "movie.en.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in result["segments"]] == [f"EN line {index}" for index in range(5)]
    assert not (output_dir / ".mlxscribe-work").exists()
