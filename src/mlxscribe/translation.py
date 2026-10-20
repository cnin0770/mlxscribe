from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mlxscribe.exporters.json_exporter import write_json
from mlxscribe.exporters.srt import write_srt
from mlxscribe.exporters.txt import write_txt

DEFAULT_TRANSLATION_MODEL = "gpt-5.5"
DEFAULT_MAX_SEGMENTS_PER_BATCH = 40


@dataclass(frozen=True)
class TokenEstimate:
    source_text_tokens: int
    prompt_tokens: int
    expected_output_tokens: int
    total_tokens: int
    tokenizer: str


def estimate_translation_tokens(segments: list[dict], *, target_language: str) -> TokenEstimate:
    source_text = "\n".join(str(segment["text"]) for segment in segments)
    request_overhead = (
        "Translate these subtitle segments. Preserve meaning, tone, names, and line order. "
        f"Target language: {target_language}."
    )
    source_tokens, tokenizer = count_tokens(source_text)
    overhead_tokens, _ = count_tokens(request_overhead)
    expected_output_tokens = max(1, int(source_tokens * output_multiplier(target_language)))
    prompt_tokens = source_tokens + overhead_tokens + len(segments) * 8
    return TokenEstimate(
        source_text_tokens=source_tokens,
        prompt_tokens=prompt_tokens,
        expected_output_tokens=expected_output_tokens,
        total_tokens=prompt_tokens + expected_output_tokens,
        tokenizer=tokenizer,
    )


def count_tokens(text: str) -> tuple[int, str]:
    try:
        import tiktoken

        try:
            encoding = tiktoken.get_encoding("o200k_base")
            return len(encoding.encode(text)), "o200k_base"
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text)), "cl100k_base"
    except Exception:
        return max(1, len(text) // 3), "char_estimate"


def output_multiplier(target_language: str) -> float:
    normalized = target_language.lower()
    if normalized.startswith("zh"):
        return 0.9
    if normalized.startswith("en"):
        return 1.4
    return 1.2


def translate_segments_openai(
    segments: list[dict],
    *,
    target_language: str,
    model: str,
    max_segments_per_batch: int,
) -> tuple[list[dict], dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed; run `uv sync` first") from exc

    client = OpenAI()
    translated: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "batches": 0}

    for batch in batched(segments, max_segments_per_batch):
        response = client.responses.create(
            model=model,
            instructions=translation_instructions(target_language),
            input=json.dumps(
                [{"id": segment["id"], "text": segment["text"]} for segment in batch],
                ensure_ascii=False,
            ),
        )
        translations = parse_translation_response(response.output_text)
        by_id = {item["id"]: item["text"] for item in translations}
        for segment in batch:
            translated_segment = {**segment, "source_text": segment["text"]}
            translated_segment["text"] = by_id.get(segment["id"], segment["text"])
            translated.append(translated_segment)
        add_usage(usage, response)

    return translated, usage


def translation_instructions(target_language: str) -> str:
    return (
        "You translate subtitle segments for film/video. "
        "Preserve segment order and IDs. Preserve names and speaker intent. "
        "Return only a JSON array. Each item must be an object with id and text. "
        f"Translate into {target_language}. Keep subtitle lines concise and natural."
    )


def parse_translation_response(text: str) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if not match:
            raise ValueError("translation response was not JSON")
        value = json.loads(match.group(0))

    if not isinstance(value, list):
        raise ValueError("translation response must be a JSON array")

    parsed = []
    for item in value:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError("each translation item must contain id and text")
        parsed.append({"id": item["id"], "text": str(item["text"]).strip()})
    return parsed


def add_usage(usage: dict, response: object) -> None:
    usage["batches"] += 1
    response_usage = getattr(response, "usage", None)
    if response_usage is None:
        return

    input_tokens = getattr(response_usage, "input_tokens", None) or 0
    output_tokens = getattr(response_usage, "output_tokens", None) or 0
    total_tokens = getattr(response_usage, "total_tokens", None) or input_tokens + output_tokens
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["total_tokens"] += total_tokens


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_translated_result(
    *,
    source: dict,
    source_language: str | None,
    target_language: str,
    translated_segments: list[dict],
    model: str | None,
    engine: str,
    token_estimate: TokenEstimate,
    usage: dict | None = None,
) -> dict:
    return {
        "text": "".join(segment["text"] for segment in translated_segments),
        "segments": translated_segments,
        "language": target_language,
        "source_language": source_language or source.get("language"),
        "translation": {
            "engine": engine,
            "model": model,
            "token_estimate": token_estimate.__dict__,
            "usage": usage,
        },
    }


def write_translation_outputs(
    *,
    result: dict,
    output_dir: Path,
    stem: str,
    target_language: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stem}.{target_language}"
    paths = {
        "json": str(output_dir / f"{base}.json"),
        "srt": str(output_dir / f"{base}.srt"),
        "txt": str(output_dir / f"{base}.txt"),
    }
    write_json(Path(paths["json"]), result)
    write_srt(Path(paths["srt"]), result)
    write_txt(Path(paths["txt"]), result)
    return paths
