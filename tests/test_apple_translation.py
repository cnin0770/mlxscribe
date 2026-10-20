import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path

import pytest

from mlxscribe import apple_translation, cli
from mlxscribe.apple_translation import AppleTranslator, check_language_pair

# Speaks the real helper's protocol: `check <src> <dst>` prints a status, and
# `translate <src> <dst>` maps a JSON array of {id, text} on stdin to one on stdout.
FAKE_HELPER = """#!{python}
import json, os, sys
mode, source, target = sys.argv[1:4]
log = os.environ.get("FAKE_HELPER_LOG")
if mode == "check":
    print(os.environ.get("FAKE_HELPER_STATUS", "installed"))
    sys.exit(0)
items = json.load(sys.stdin)
if log:
    with open(log, "a") as handle:
        handle.write(json.dumps([item["id"] for item in items]) + "\\n")
fail_on = {{int(value) for value in os.environ.get("FAKE_HELPER_FAIL_ON", "").split(",") if value}}
if fail_on & {{item["id"] for item in items}}:
    print("Error: language pack busy", file=sys.stderr)
    sys.exit(1)
json.dump([{{"id": item["id"], "text": f"{{target}}:{{item['text']}}"}} for item in items], sys.stdout)
"""

SEGMENTS = [
    {"id": index, "start": float(index), "end": index + 1.0, "text": f"台詞{index}"} for index in range(5)
]


@pytest.fixture
def fake_helper(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fake-helper"
    path.write_text(FAKE_HELPER.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(apple_translation, "ensure_helper", lambda: path)
    monkeypatch.setenv("FAKE_HELPER_LOG", str(tmp_path / "calls.log"))
    return path


def helper_calls(tmp_path: Path) -> list[list[int]]:
    log = tmp_path / "calls.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_apple_translator_skips_blank_segments(fake_helper) -> None:
    translator = AppleTranslator("ja", "zh-Hans")

    translations, usage = translator(
        [{"id": 0, "text": " こんにちは "}, {"id": 1, "text": "  "}, {"id": 2, "text": "はい"}]
    )

    assert translations == [{"id": 0, "text": "zh-Hans:こんにちは"}, {"id": 2, "text": "zh-Hans:はい"}]
    assert usage == {"batches": 1}


def test_apple_translator_maps_bare_chinese_source_code(fake_helper) -> None:
    assert AppleTranslator("zh", "en").source == "zh-Hans"


def test_apple_translator_reports_helper_errors(fake_helper, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_FAIL_ON", "0")

    with pytest.raises(RuntimeError, match="language pack busy"):
        AppleTranslator("ja", "en")([{"id": 0, "text": "はい"}])


def test_check_language_pair_explains_missing_language_pack(fake_helper, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_STATUS", "supported")
    with pytest.raises(RuntimeError, match="Translation Languages"):
        check_language_pair("ja", "zh-Hans")

    monkeypatch.setenv("FAKE_HELPER_STATUS", "unsupported")
    with pytest.raises(RuntimeError, match="does not support ja -> xx"):
        check_language_pair("ja", "xx")

    monkeypatch.setenv("FAKE_HELPER_STATUS", "installed")
    assert check_language_pair("ja", "zh-Hans") == "installed"


def translate_command(source: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "translate", str(source), "--to", "zh-Hans", "--engine", "apple", "--output", str(output_dir),
        "--batch-size", "2", "--max-attempts", "1", "--retry-delay", "0", *extra,
    ]


def test_translate_command_with_apple_engine_writes_outputs(tmp_path, fake_helper, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "movie.ja.json"
    source.write_text(json.dumps({"language": "ja", "segments": SEGMENTS}, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "out"

    assert cli.main(translate_command(source, output_dir)) == 0

    out = capsys.readouterr().out
    assert "Model: Apple Translation (macOS" in out
    assert "tokens" not in out
    result = json.loads((output_dir / "movie.zh-Hans.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in result["segments"]] == [f"zh-Hans:台詞{i}" for i in range(5)]
    assert [segment["source_text"] for segment in result["segments"]] == [f"台詞{i}" for i in range(5)]
    assert result["translation"]["engine"] == "apple"
    assert result["translation"]["model"].startswith("macOS ")
    assert "zh-Hans:台詞0" in (output_dir / "movie.zh-Hans.srt").read_text(encoding="utf-8")
    assert helper_calls(tmp_path) == [[0, 1], [2, 3], [4]]


def test_translate_command_with_apple_engine_resumes(tmp_path, fake_helper, monkeypatch, capsys) -> None:
    source = tmp_path / "movie.ja.json"
    source.write_text(json.dumps({"language": "ja", "segments": SEGMENTS}, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "out"

    monkeypatch.setenv("FAKE_HELPER_FAIL_ON", "2")
    assert cli.main(translate_command(source, output_dir)) == 1
    assert "rerun the same command to resume" in capsys.readouterr().err

    monkeypatch.delenv("FAKE_HELPER_FAIL_ON")
    assert cli.main(translate_command(source, output_dir)) == 0
    assert "Resumed batches: 1/3" in capsys.readouterr().out
    # First run: batch 0 done, batch 1 failed. Second run: only batches 1 and 2.
    assert helper_calls(tmp_path) == [[0, 1], [2, 3], [2, 3], [4]]
    assert not (output_dir / ".mlxscribe-work").exists()


def test_translate_command_with_apple_engine_stops_without_language_pack(
    tmp_path, fake_helper, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FAKE_HELPER_STATUS", "supported")
    source = tmp_path / "movie.ja.json"
    source.write_text(json.dumps({"language": "ja", "segments": SEGMENTS}, ensure_ascii=False), encoding="utf-8")

    assert cli.main(translate_command(source, tmp_path / "out")) == 1

    assert "System Settings" in capsys.readouterr().err
    assert helper_calls(tmp_path) == []


@pytest.mark.skipif(
    os.environ.get("MLXSCRIBE_APPLE_TESTS") != "1"
    or platform.system() != "Darwin"
    or not shutil.which("swiftc"),
    reason="set MLXSCRIBE_APPLE_TESTS=1 on a Mac with swiftc and the ja -> en language pack",
)
def test_real_helper_translates_japanese(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLXSCRIBE_CACHE_DIR", str(tmp_path / "cache"))

    helper = apple_translation.ensure_helper()
    check_language_pair("ja", "en", helper=helper)
    translations, _ = AppleTranslator("ja", "en", helper=helper)(
        [{"id": 7, "text": "ありがとうございます"}]
    )

    assert translations[0]["id"] == 7
    assert "thank" in translations[0]["text"].lower()
