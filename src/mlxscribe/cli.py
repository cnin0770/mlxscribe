from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from mlxscribe import __version__
from mlxscribe.benchmark import print_benchmark_summary
from mlxscribe.chunking import (
    Chunk,
    merge_chunk_results,
    offset_result,
    plan_chunks,
    slice_audio,
    snap_chunks_to_quiet,
)
from mlxscribe.engine import transcribe_audio
from mlxscribe.exporters.json_exporter import write_json
from mlxscribe.exporters.srt import write_srt
from mlxscribe.exporters.txt import write_txt
from mlxscribe.filters import FilterOptions, filter_transcript
from mlxscribe.formats import parse_formats
from mlxscribe.media import extract_audio, load_wav, probe_duration
from mlxscribe.transcripts import clean_segments, load_transcript, transcript_stem
from mlxscribe.translation import (
    DEFAULT_MAX_SEGMENTS_PER_BATCH,
    DEFAULT_TRANSLATION_MODEL,
    build_translated_result,
    estimate_translation_tokens,
    translate_segments_openai,
    write_translation_outputs,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FORMATS = ("srt", "json", "txt")
AUTO_LANGUAGE = "auto"
CONTEXT_AUTO = "auto"
LONG_FORM_THRESHOLD_SECONDS = 15 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = PROJECT_ROOT / ".hf-cache"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            return check_environment()
        if args.command == "transcribe":
            return run_transcribe(args)
        if args.command == "summary":
            return run_summary(args)
        if args.command == "translate":
            return run_translate(args)
        if args.command == "filter":
            return run_filter(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxscribe",
        description="Local-first transcription CLI powered by mlx-whisper.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="Check required local tools.")
    check_parser.set_defaults(command="check")

    summary_parser = subparsers.add_parser(
        "summary",
        help="Print a compact table for benchmark JSON files.",
    )
    summary_parser.add_argument(
        "benchmarks",
        nargs="+",
        type=Path,
        help="One or more *.benchmark.json files.",
    )
    summary_parser.set_defaults(command="summary")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Extract audio from a local media file and transcribe it.",
    )
    transcribe_parser.add_argument("input", type=Path, help="Local audio or video file.")
    transcribe_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where transcript files will be written.",
    )
    transcribe_parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help="Source language code, or auto for Whisper language detection. Defaults to auto.",
    )
    transcribe_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX Whisper model path or Hugging Face repo. Defaults to {DEFAULT_MODEL}.",
    )
    transcribe_parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated output formats: srt,json,txt.",
    )
    transcribe_parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper task. Translation target is English when supported by the model.",
    )
    transcribe_parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Ask mlx-whisper for word-level timestamps.",
    )
    transcribe_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decode temperature. Defaults to 0.0 for predictable, fast decoding.",
    )
    transcribe_parser.add_argument(
        "--context",
        choices=(CONTEXT_AUTO, "on", "off"),
        default=CONTEXT_AUTO,
        help="Previous-text context mode. Auto disables context above 15 minutes.",
    )
    transcribe_parser.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text_override",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    transcribe_parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text_override",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    transcribe_parser.add_argument(
        "--verbose-engine",
        action="store_true",
        help="Print mlx-whisper segment output while transcribing.",
    )
    transcribe_parser.add_argument(
        "--keep-audio",
        type=Path,
        default=None,
        metavar="DIR",
        help="Keep the extracted 16 kHz mono WAV in this directory.",
    )
    transcribe_parser.add_argument(
        "--filter-noise",
        action="store_true",
        help="Apply conservative transcript noise filtering before writing outputs.",
    )
    transcribe_parser.add_argument(
        "--filter-mode",
        choices=("conservative",),
        default="conservative",
        help="Noise filtering mode.",
    )
    transcribe_parser.add_argument(
        "--chunk-minutes",
        type=positive_float,
        default=None,
        help="Transcribe in sequential chunks of this many minutes.",
    )
    transcribe_parser.set_defaults(command="transcribe")

    filter_parser = subparsers.add_parser(
        "filter",
        help="Apply conservative noise filtering to an existing transcript JSON.",
    )
    filter_parser.add_argument("input", type=Path, help="Transcript JSON from mlxscribe.")
    filter_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where filtered files will be written.",
    )
    filter_parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated output formats: srt,json,txt.",
    )
    filter_parser.add_argument(
        "--filter-mode",
        choices=("conservative",),
        default="conservative",
        help="Noise filtering mode.",
    )
    filter_parser.set_defaults(command="filter")

    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate a transcript JSON while preserving segment timestamps.",
    )
    translate_parser.add_argument("input", type=Path, help="Transcript JSON from mlxscribe.")
    translate_parser.add_argument(
        "--to",
        required=True,
        dest="target_language",
        help="Target language, for example en, zh-Hans, or zh-Hant.",
    )
    translate_parser.add_argument(
        "--from",
        dest="source_language",
        default=None,
        help="Source language override. Defaults to the transcript language.",
    )
    translate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where translated files will be written.",
    )
    translate_parser.add_argument(
        "--engine",
        choices=("estimate", "openai"),
        default="estimate",
        help="Translation engine. Defaults to estimate, which spends no API tokens.",
    )
    translate_parser.add_argument(
        "--model",
        default=DEFAULT_TRANSLATION_MODEL,
        help=f"OpenAI model for --engine openai. Defaults to {DEFAULT_TRANSLATION_MODEL}.",
    )
    translate_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_MAX_SEGMENTS_PER_BATCH,
        help="Number of subtitle segments to translate per request.",
    )
    translate_parser.set_defaults(command="translate")

    return parser


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def check_environment() -> int:
    tools = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }

    ok = True
    for name, path in tools.items():
        if path:
            print(f"{name}: {path}")
        else:
            print(f"{name}: missing")
            ok = False

    try:
        version = importlib.metadata.version("mlx-whisper")
        print(f"mlx-whisper: installed ({version})")
    except importlib.metadata.PackageNotFoundError:
        print("mlx-whisper: missing")
        ok = False

    hf_home = get_hf_home()
    print(f"HF_HOME: {hf_home}")
    if hf_home.exists():
        print("HF cache: present")
    else:
        print("HF cache: will be created on first model download")

    return 0 if ok else 1


def run_transcribe(args: argparse.Namespace) -> int:
    configure_huggingface_cache()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    formats = parse_formats(args.formats)

    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    started_at = time.perf_counter()
    phase_timings: dict[str, float] = {}
    phase_started_at = time.perf_counter()
    source_duration = probe_duration(input_path)
    phase_timings["probe_duration_seconds"] = time.perf_counter() - phase_started_at

    print(f"Input: {input_path}")
    if source_duration is not None:
        print(f"Duration: {source_duration:.2f}s")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    chunks = plan_chunks(source_duration, args.chunk_minutes)
    context_decision = resolve_context_mode(
        requested_mode=args.context,
        duration_seconds=source_duration,
        explicit_override=args.condition_on_previous_text_override,
    )
    args.condition_on_previous_text = context_decision["enabled"]
    print(
        "Context: "
        f"{'on' if context_decision['enabled'] else 'off'} "
        f"({context_decision['reason']})"
    )
    if chunks:
        print(f"Chunks: {len(chunks)} x {args.chunk_minutes:g} minute(s)")
    elif args.chunk_minutes:
        print("Chunks: disabled (source duration unknown)")
    print(f"HF_HOME: {get_hf_home()}")
    print("Extracting audio...")

    kept_audio_path = None
    if args.keep_audio is not None:
        keep_audio_dir = args.keep_audio.expanduser().resolve()
        keep_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = keep_audio_dir / f"{stem}.16k-mono.wav"
        phase_started_at = time.perf_counter()
        extract_audio(input_path, audio_path)
        phase_timings["extract_audio_seconds"] = time.perf_counter() - phase_started_at
        kept_audio_path = audio_path
        result = run_transcription(args, audio_path, chunks)
        phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")
    else:
        with tempfile.TemporaryDirectory(prefix="mlxscribe-") as temp_dir:
            audio_path = Path(temp_dir) / f"{stem}.wav"
            phase_started_at = time.perf_counter()
            extract_audio(input_path, audio_path)
            phase_timings["extract_audio_seconds"] = time.perf_counter() - phase_started_at
            result = run_transcription(args, audio_path, chunks)
            phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")

    filter_summary = None
    if args.filter_noise:
        result, filter_summary = filter_transcript(
            result,
            FilterOptions(enabled=True, mode=args.filter_mode),
        )

    output_language = result.get("language") or args.language
    phase_started_at = time.perf_counter()
    write_outputs(
        result=result,
        output_dir=output_dir,
        stem=stem,
        language=output_language,
        formats=formats,
    )
    phase_timings["export_seconds"] = time.perf_counter() - phase_started_at
    elapsed = time.perf_counter() - started_at
    phase_timings["total_seconds"] = elapsed

    benchmark = {
        "input": str(input_path),
        "model": args.model,
        "language": output_language,
        "requested_language": args.language,
        "task": args.task,
        "formats": formats,
        "word_timestamps": args.word_timestamps,
        "temperature": args.temperature,
        "condition_on_previous_text": args.condition_on_previous_text,
        "condition_on_previous_text_requested": context_decision["requested"],
        "condition_on_previous_text_reason": context_decision["reason"],
        "long_form_threshold_seconds": LONG_FORM_THRESHOLD_SECONDS,
        "chunking": {
            "enabled": bool(chunks),
            "chunk_minutes": args.chunk_minutes,
            "chunks": result.get("chunks", []),
        },
        "hf_home": str(get_hf_home()),
        "kept_audio": str(kept_audio_path) if kept_audio_path else None,
        "source_duration_seconds": source_duration,
        "elapsed_seconds": elapsed,
        "realtime_factor": (source_duration / elapsed) if source_duration else None,
        "phase_timings": phase_timings,
        "filter": filter_summary or {
            "enabled": False,
            "mode": args.filter_mode,
            "dropped_segments": 0,
            "dropped": [],
            "rules": [],
        },
    }
    benchmark_path = output_dir / f"{stem}.benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {output_dir}")
    if source_duration:
        print(f"Elapsed: {elapsed:.2f}s ({source_duration / elapsed:.2f}x realtime)")
    else:
        print(f"Elapsed: {elapsed:.2f}s")
    print_phase_timings(phase_timings)
    if filter_summary is not None:
        print(
            "Filter: "
            f"dropped {filter_summary['dropped_segments']} segment(s), "
            f"kept {filter_summary['kept_segments']}"
        )
    return 0


def run_summary(args: argparse.Namespace) -> int:
    paths = [path.expanduser().resolve() for path in args.benchmarks]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"benchmark file does not exist: {path}")
    print_benchmark_summary(paths)
    return 0


def run_translate(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    source = load_transcript(input_path)
    source_language = args.source_language or source.get("language")
    segments = clean_segments(source)
    if not segments:
        raise ValueError("transcript contains no translatable segments")

    estimate = estimate_translation_tokens(segments, target_language=args.target_language)
    batches = (len(segments) + args.batch_size - 1) // args.batch_size

    print(f"Input: {input_path}")
    print(f"Segments: {len(segments)}")
    print(f"Target: {args.target_language}")
    print(f"Engine: {args.engine}")
    print(f"Estimated prompt tokens: {estimate.prompt_tokens}")
    print(f"Estimated output tokens: {estimate.expected_output_tokens}")
    print(f"Estimated total tokens: {estimate.total_tokens}")
    print(f"Tokenizer: {estimate.tokenizer}")
    print(f"Batches: {batches}")

    stem = transcript_stem(input_path, source_language)
    if args.engine == "estimate":
        estimate_result = build_translated_result(
            source=source,
            source_language=source_language,
            target_language=args.target_language,
            translated_segments=segments,
            model=None,
            engine="estimate",
            token_estimate=estimate,
            usage=None,
        )
        estimate_path = output_dir / f"{stem}.{args.target_language}.estimate.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        estimate_path.write_text(json.dumps(estimate_result["translation"], indent=2), encoding="utf-8")
        print(f"Wrote estimate to {estimate_path}")
        print("No API call was made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --engine openai")

    started_at = time.perf_counter()
    translated_segments, usage = translate_segments_openai(
        segments,
        target_language=args.target_language,
        model=args.model,
        max_segments_per_batch=args.batch_size,
    )
    elapsed = time.perf_counter() - started_at
    result = build_translated_result(
        source=source,
        source_language=source_language,
        target_language=args.target_language,
        translated_segments=translated_segments,
        model=args.model,
        engine="openai",
        token_estimate=estimate,
        usage=usage,
    )
    paths = write_translation_outputs(
        result=result,
        output_dir=output_dir,
        stem=stem,
        target_language=args.target_language,
    )

    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Actual input tokens: {usage.get('input_tokens', 0)}")
    print(f"Actual output tokens: {usage.get('output_tokens', 0)}")
    print(f"Actual total tokens: {usage.get('total_tokens', 0)}")
    print(f"Wrote JSON: {paths['json']}")
    print(f"Wrote SRT: {paths['srt']}")
    print(f"Wrote TXT: {paths['txt']}")
    return 0


def run_filter(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    formats = parse_formats(args.formats)
    source = load_transcript(input_path)
    source_language = source.get("language") or "unknown"
    stem = transcript_stem(input_path, source_language)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered, summary = filter_transcript(
        source,
        FilterOptions(enabled=True, mode=args.filter_mode),
    )
    write_outputs(
        result=filtered,
        output_dir=output_dir,
        stem=stem,
        language=source_language,
        formats=formats,
    )

    print(f"Input: {input_path}")
    print(f"Wrote filtered outputs to {output_dir}")
    print(
        "Filter: "
        f"dropped {summary['dropped_segments']} segment(s), "
        f"kept {summary['kept_segments']}"
    )
    if summary["rules"]:
        print(f"Rules: {', '.join(summary['rules'])}")
    return 0


def run_transcription(args: argparse.Namespace, audio_path: Path, chunks: list[Chunk]) -> dict:
    if chunks:
        return run_chunked_engine(args, audio_path, chunks)
    return run_engine(args, audio_path)


def run_engine(args: argparse.Namespace, audio_path: Path) -> dict:
    print("Transcribing...")
    started_at = time.perf_counter()
    language = normalize_language(args.language)
    result = transcribe_audio(
        audio_path,
        model=args.model,
        language=language,
        task=args.task,
        word_timestamps=args.word_timestamps,
        temperature=args.temperature,
        condition_on_previous_text=args.condition_on_previous_text,
        verbose=args.verbose_engine,
    )
    result["_elapsed_seconds"] = time.perf_counter() - started_at
    return result


def run_chunked_engine(args: argparse.Namespace, audio_path: Path, chunks: list[Chunk]) -> dict:
    started_at = time.perf_counter()
    audio = load_wav(audio_path)
    planned_ends = [chunk.end for chunk in chunks]
    chunks = snap_chunks_to_quiet(chunks, audio)
    print("Transcribing chunks...")
    language = normalize_language(args.language)
    results = []
    chunk_timings = []
    for chunk in chunks:
        chunk_started_at = time.perf_counter()
        print(
            f"  chunk {chunk.index + 1}/{len(chunks)}: "
            f"{format_seconds(chunk.start)}-{format_seconds(chunk.end)}"
        )
        result = transcribe_audio(
            slice_audio(audio, chunk),
            model=args.model,
            language=language,
            task=args.task,
            word_timestamps=args.word_timestamps,
            temperature=args.temperature,
            condition_on_previous_text=args.condition_on_previous_text,
            verbose=args.verbose_engine,
        )
        result = offset_result(result, chunk.start)
        elapsed = time.perf_counter() - chunk_started_at
        results.append(result)
        chunk_timings.append(
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "elapsed_seconds": elapsed,
                "segments": len(result.get("segments") or []),
            }
        )

    merged = merge_chunk_results(results, chunks)
    for summary, timing, planned_end in zip(
        merged["chunks"], chunk_timings, planned_ends, strict=True
    ):
        summary["elapsed_seconds"] = timing["elapsed_seconds"]
        summary["planned_end"] = planned_end
    merged["_elapsed_seconds"] = time.perf_counter() - started_at
    return merged


def normalize_language(language: str) -> str | None:
    normalized = language.strip()
    if normalized.lower() == AUTO_LANGUAGE:
        return None
    return normalized


def resolve_context_mode(
    *,
    requested_mode: str,
    duration_seconds: float | None,
    explicit_override: bool | None = None,
) -> dict[str, object]:
    if explicit_override is not None:
        return {
            "enabled": explicit_override,
            "requested": "on" if explicit_override else "off",
            "reason": "explicit legacy override",
        }

    if requested_mode == "on":
        return {
            "enabled": True,
            "requested": requested_mode,
            "reason": "explicit context mode",
        }
    if requested_mode == "off":
        return {
            "enabled": False,
            "requested": requested_mode,
            "reason": "explicit context mode",
        }
    if requested_mode != CONTEXT_AUTO:
        raise ValueError(f"unsupported context mode: {requested_mode}")

    if duration_seconds is not None and duration_seconds > LONG_FORM_THRESHOLD_SECONDS:
        return {
            "enabled": False,
            "requested": requested_mode,
            "reason": f"source duration exceeded {LONG_FORM_THRESHOLD_SECONDS}s",
        }

    return {
        "enabled": True,
        "requested": requested_mode,
        "reason": f"source duration did not exceed {LONG_FORM_THRESHOLD_SECONDS}s",
    }


def print_phase_timings(phase_timings: dict[str, float]) -> None:
    print("Phase timings:")
    for name, seconds in phase_timings.items():
        label = name.removesuffix("_seconds").replace("_", " ")
        print(f"  {label}: {seconds:.2f}s")


def format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def configure_huggingface_cache() -> Path:
    hf_home = get_hf_home()
    os.environ.setdefault("HF_HOME", str(hf_home))
    hf_home.mkdir(parents=True, exist_ok=True)
    return hf_home


def get_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser().resolve()


def write_outputs(
    *,
    result: dict,
    output_dir: Path,
    stem: str,
    language: str,
    formats: tuple[str, ...],
) -> None:
    base = output_dir / f"{stem}.{language}"
    if "json" in formats:
        write_json(base.with_suffix(".json"), result)
    if "txt" in formats:
        write_txt(base.with_suffix(".txt"), result)
    if "srt" in formats:
        write_srt(base.with_suffix(".srt"), result)


if __name__ == "__main__":
    raise SystemExit(main())
