from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from mlxscribe import __version__, apple_translation
from mlxscribe.benchmark import print_benchmark_summary
from mlxscribe.chunking import (
    Chunk,
    QualityOptions,
    assess_chunk,
    merge_chunk_results,
    offset_result,
    plan_chunks,
    quality_rank,
    retry_temperatures,
    slice_audio,
    snap_chunks_to_quiet,
)
from mlxscribe.engine import transcribe_audio
from mlxscribe.exporters.json_exporter import write_json
from mlxscribe.exporters.srt import write_srt
from mlxscribe.exporters.txt import write_txt
from mlxscribe.filters import FilterOptions, filter_transcript
from mlxscribe.formats import parse_formats
from mlxscribe.jobs import Checkpoint, ProgressWriter, file_fingerprint, work_paths
from mlxscribe.media import extract_audio, load_wav, probe_duration, wav_duration
from mlxscribe.progress_display import JobProgress, whisper_progress
from mlxscribe.subtitles import SubtitleOptions, clean_subtitles
from mlxscribe.transcripts import clean_segments, load_transcript, transcript_stem
from mlxscribe.translation import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_SEGMENTS_PER_BATCH,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_TRANSLATION_MODEL,
    build_translated_result,
    estimate_translation_tokens,
    translate_segments,
    translate_segments_openai,
    translation_fingerprint,
    write_translation_outputs,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FORMATS = ("srt", "json", "txt")
AUTO_LANGUAGE = "auto"
CONTEXT_AUTO = "auto"
LONG_FORM_THRESHOLD_SECONDS = 15 * 60
CHUNK_AUTO = "auto"
CHUNK_OFF = "off"
DEFAULT_CHUNK_MINUTES = 5.0
DEFAULT_SKIP_SILENCE_SECONDS = 2.0
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
        "--skip-silence",
        type=parse_skip_silence,
        default=DEFAULT_SKIP_SILENCE_SECONDS,
        metavar="SECONDS",
        help=(
            "Skip silent or noise-only gaps longer than SECONDS, and the text Whisper invents "
            "over them, so subtitles start when speech starts. Turns on --word-timestamps and "
            "roughly doubles transcription time. 'off' is the fast path. "
            f"Defaults to {DEFAULT_SKIP_SILENCE_SECONDS:g}."
        ),
    )
    transcribe_parser.add_argument(
        "--temperature",
        type=parse_temperature,
        default=0.0,
        help=(
            "Decode temperature, or a comma-separated fallback list such as 0,0.2,0.4 "
            "that mlx-whisper steps through when a 30s window looks repetitive. "
            "Defaults to 0.0."
        ),
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
        type=parse_chunk_minutes,
        default=CHUNK_AUTO,
        help=(
            "Transcribe in sequential chunks of this many minutes. 'auto' (default) "
            f"uses {DEFAULT_CHUNK_MINUTES:g}-minute chunks for inputs longer than "
            f"{LONG_FORM_THRESHOLD_SECONDS // 60} minutes; 'off' always transcribes "
            "the whole file at once."
        ),
    )
    transcribe_parser.add_argument(
        "--chunk-retries",
        type=non_negative_int,
        default=1,
        help=(
            "Retry a chunk flagged as looping or empty this many times with a higher "
            "temperature. Defaults to 1; 0 disables retries."
        ),
    )
    add_subtitle_arguments(transcribe_parser)
    add_job_arguments(transcribe_parser, unit="chunk", plural="chunks")
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
    add_subtitle_arguments(filter_parser)
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
        choices=("estimate", "openai", "apple"),
        default="estimate",
        help=(
            "Translation engine: estimate (default) counts tokens without translating, "
            "openai uses the OpenAI API, apple uses macOS on-device translation (free, offline)."
        ),
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
    translate_parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "Attempts per batch before giving up, with exponential backoff between them. "
            f"Defaults to {DEFAULT_MAX_ATTEMPTS}."
        ),
    )
    translate_parser.add_argument(
        "--retry-delay",
        type=non_negative_float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help=(
            "Seconds to wait before the first retry; doubles after each failure. "
            f"Defaults to {DEFAULT_RETRY_DELAY_SECONDS:g}."
        ),
    )
    add_subtitle_arguments(translate_parser)
    add_job_arguments(translate_parser, unit="batch", plural="batches")
    translate_parser.set_defaults(command="translate")

    return parser


def add_job_arguments(parser: argparse.ArgumentParser, *, unit: str, plural: str) -> None:
    group = parser.add_argument_group(
        "resume",
        f"Each finished {unit} is saved to a work directory so an interrupted run can resume.",
    )
    group.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"Reuse {plural} saved by an earlier run with the same input and settings. "
            "--no-resume discards them and starts fresh. Defaults to resume."
        ),
    )
    group.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the work directory after a successful run instead of deleting it.",
    )
    group.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Work directory. Defaults to <output>/.mlxscribe-work/<input name>.",
    )
    group.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        metavar="FILE",
        help=f"Append JSON-lines progress events (job, phase, {unit}) to this file.",
    )


def add_subtitle_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = SubtitleOptions()
    group = parser.add_argument_group(
        "subtitle cleanup",
        "Applied to SRT and TXT output only; JSON keeps the raw segments.",
    )
    group.add_argument(
        "--no-subtitle-cleanup",
        dest="subtitle_cleanup",
        action="store_false",
        help="Write each segment as one cue, unwrapped, as before cleanup existed.",
    )
    group.add_argument(
        "--max-lines",
        type=positive_int,
        default=defaults.max_lines,
        help=f"Maximum lines per subtitle. Defaults to {defaults.max_lines}.",
    )
    group.add_argument(
        "--max-line-chars",
        type=positive_int,
        default=defaults.max_line_chars,
        help=(
            "Maximum characters per line for Latin-script text. "
            f"Defaults to {defaults.max_line_chars}."
        ),
    )
    group.add_argument(
        "--max-line-chars-cjk",
        type=positive_int,
        default=defaults.max_line_chars_cjk,
        help=(
            "Maximum characters per line for Chinese, Japanese, or Korean text. "
            f"Defaults to {defaults.max_line_chars_cjk}."
        ),
    )
    group.add_argument(
        "--min-cue-seconds",
        type=non_negative_float,
        default=defaults.min_duration_seconds,
        help=(
            "Minimum time a subtitle stays on screen, never overlapping the next one. "
            f"Defaults to {defaults.min_duration_seconds:g}."
        ),
    )
    group.add_argument(
        "--max-cue-seconds",
        type=positive_float,
        default=defaults.max_duration_seconds,
        help=(
            "Split subtitles longer than this into timed pieces. "
            f"Defaults to {defaults.max_duration_seconds:g}."
        ),
    )
    group.add_argument(
        "--merge-gap-seconds",
        type=non_negative_float,
        default=defaults.merge_gap_seconds,
        help=(
            "Merge a short subtitle into its neighbour when the gap between them is at "
            f"most this long and the result still fits. Defaults to {defaults.merge_gap_seconds:g}."
        ),
    )


def subtitle_options_from_args(args: argparse.Namespace) -> SubtitleOptions:
    return SubtitleOptions(
        enabled=args.subtitle_cleanup,
        max_lines=args.max_lines,
        max_line_chars=args.max_line_chars,
        max_line_chars_cjk=args.max_line_chars_cjk,
        min_duration_seconds=args.min_cue_seconds,
        max_duration_seconds=args.max_cue_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
    )


def print_subtitle_summary(summary: dict) -> None:
    if not summary.get("enabled"):
        return
    print(
        "Subtitles: "
        f"{summary['input_segments']} segment(s) -> {summary['output_cues']} cue(s); "
        f"split {summary['split_segments']}, merged {summary['merged_segments']}, "
        f"extended {summary['extended_cues']}, wrapped {summary['wrapped_cues']}"
    )
    remaining = {
        "long lines": summary["remaining_long_lines"],
        "short cues": summary["remaining_short_cues"],
        "cues over max duration": summary["remaining_long_cues"],
        "overlaps": summary["remaining_overlaps"],
    }
    left = [f"{count} {label}" for label, count in remaining.items() if count]
    if left:
        print(f"Subtitles left as is: {', '.join(left)}")


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def parse_temperature(value: str) -> float | tuple[float, ...]:
    try:
        values = tuple(float(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected numbers, got {value!r}") from None
    if any(not 0 <= number <= 1 for number in values):
        raise argparse.ArgumentTypeError(f"temperatures must be between 0 and 1, got {value}")
    return values[0] if len(values) == 1 else values


def parse_skip_silence(value: str) -> float | None:
    if value.strip().lower() == "off":
        return None
    return positive_float(value)


def parse_chunk_minutes(value: str) -> str | float:
    normalized = value.strip().lower()
    if normalized in (CHUNK_AUTO, CHUNK_OFF):
        return normalized
    return positive_float(value)


def non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or greater, got {value}")
    return number


def positive_int(value: str) -> int:
    number = non_negative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or greater, got {value}")
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

    # Optional: only `translate --engine apple` needs it, so it never fails the check.
    print(f"Apple Translation: {describe_apple_translation()}")

    return 0 if ok else 1


def describe_apple_translation() -> str:
    if apple_translation.helper_path().exists():
        return f"helper built ({apple_translation.helper_path()})"
    if shutil.which("swiftc"):
        return "available; the helper is built on first use"
    return "unavailable (needs swiftc: `xcode-select --install`)"


def run_transcribe(args: argparse.Namespace) -> int:
    if args.skip_silence is not None:
        args.word_timestamps = True
    progress = ProgressWriter(args.progress_jsonl)
    progress.emit("job", kind="transcribe", status="started", input=str(args.input))
    try:
        return transcribe_job(args, progress)
    except BaseException as exc:
        progress.emit("job", kind="transcribe", status=job_failure_status(exc), error=str(exc))
        raise
    finally:
        progress.close()


def job_failure_status(exc: BaseException) -> str:
    return "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"


def transcribe_job(args: argparse.Namespace, progress: ProgressWriter) -> int:
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
    chunk_decision = resolve_chunk_mode(
        requested=args.chunk_minutes,
        duration_seconds=source_duration,
    )
    chunks = plan_chunks(source_duration, chunk_decision["minutes"])
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
        print(
            f"Chunks: {len(chunks)} x {chunk_decision['minutes']:g} minute(s) "
            f"({chunk_decision['reason']})"
        )
    else:
        print(f"Chunks: off ({chunk_decision['reason']})")
    print(f"HF_HOME: {get_hf_home()}")
    print("Extracting audio...")
    progress.emit("phase", phase="extract_audio", status="started")
    job = {
        "input_path": input_path,
        "output_dir": output_dir,
        "stem": stem,
        "progress": progress,
        "duration": source_duration,
    }

    kept_audio_path = None
    if args.keep_audio is not None:
        keep_audio_dir = args.keep_audio.expanduser().resolve()
        keep_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = keep_audio_dir / f"{stem}.16k-mono.wav"
        phase_started_at = time.perf_counter()
        extract_audio(input_path, audio_path)
        phase_timings["extract_audio_seconds"] = time.perf_counter() - phase_started_at
        kept_audio_path = audio_path
        result = run_transcription(args, audio_path, chunks, job)
        phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")
    else:
        with tempfile.TemporaryDirectory(prefix="mlxscribe-") as temp_dir:
            audio_path = Path(temp_dir) / f"{stem}.wav"
            phase_started_at = time.perf_counter()
            extract_audio(input_path, audio_path)
            phase_timings["extract_audio_seconds"] = time.perf_counter() - phase_started_at
            result = run_transcription(args, audio_path, chunks, job)
            phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")

    checkpoint, prune = result.pop("_checkpoint", (None, ()))
    progress.emit("phase", phase="transcribe", status="done")

    filter_summary = None
    if args.filter_noise:
        result, filter_summary = filter_transcript(
            result,
            FilterOptions(enabled=True, mode=args.filter_mode),
        )

    output_language = result.get("language") or args.language
    phase_started_at = time.perf_counter()
    subtitles, subtitle_summary = prepare_subtitles(
        result, subtitle_options_from_args(args), formats
    )
    write_outputs(
        result=result,
        subtitles=subtitles,
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
        "skip_silence": args.skip_silence,
        "temperature": args.temperature,
        "condition_on_previous_text": args.condition_on_previous_text,
        "condition_on_previous_text_requested": context_decision["requested"],
        "condition_on_previous_text_reason": context_decision["reason"],
        "long_form_threshold_seconds": LONG_FORM_THRESHOLD_SECONDS,
        "chunking": {
            "enabled": bool(chunks),
            "requested": args.chunk_minutes,
            "reason": chunk_decision["reason"],
            "chunk_minutes": chunk_decision["minutes"],
            "chunk_retries": args.chunk_retries,
            "flagged_chunks": sum(
                chunk["quality"]["flagged"] for chunk in result.get("chunks", [])
            ),
            "retried_chunks": sum(
                len(chunk["quality"]["attempts"]) > 1 for chunk in result.get("chunks", [])
            ),
            "resumed_chunks": sum(chunk.get("resumed", False) for chunk in result.get("chunks", [])),
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
        "subtitles": subtitle_summary,
    }
    benchmark_path = output_dir / f"{stem}.benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    if checkpoint is not None:
        checkpoint.finish(keep=args.keep_work, prune=prune)
    progress.emit("job", kind="transcribe", status="complete", output_dir=str(output_dir))

    print(f"Wrote outputs to {output_dir}")
    if source_duration:
        print(f"Elapsed: {elapsed:.2f}s ({source_duration / elapsed:.2f}x realtime)")
    else:
        print(f"Elapsed: {elapsed:.2f}s")
    print_phase_timings(phase_timings)
    if chunks:
        print(
            "Chunk quality: "
            f"{benchmark['chunking']['flagged_chunks']} flagged, "
            f"{benchmark['chunking']['retried_chunks']} retried"
        )
    if filter_summary is not None:
        print(
            "Filter: "
            f"dropped {filter_summary['dropped_segments']} segment(s), "
            f"kept {filter_summary['kept_segments']}"
        )
    print_subtitle_summary(subtitle_summary)
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
    if args.engine != "apple":
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

    if args.engine == "apple":
        if not source_language:
            raise ValueError("the transcript has no language; pass --from, for example --from ja")
        translator = apple_translation.AppleTranslator(source_language, args.target_language)
        apple_translation.check_language_pair(
            source_language, args.target_language, helper=translator.helper
        )
        model = f"macOS {apple_translation.macos_version()}"
        print(f"Model: Apple Translation ({model})")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --engine openai")
        model = args.model

    work_path, prune = work_paths(output_dir, stem, f"translate.{args.target_language}", args.work_dir)
    checkpoint, message = Checkpoint.open(
        work_path,
        kind="translate",
        fingerprint=translation_fingerprint(
            segments,
            source_language=source_language,
            target_language=args.target_language,
            model=model,
            max_segments_per_batch=args.batch_size,
            engine=args.engine,
        ),
        total=batches,
        resume=args.resume,
    )
    if message:
        print(message)
    progress = ProgressWriter(args.progress_jsonl)
    progress.emit("job", kind="translate", status="started", input=str(input_path), work_dir=str(work_path))
    started_at = time.perf_counter()
    display = JobProgress(len(segments), desc="Translating", unit="segments")
    try:
        if args.engine == "apple":
            translated_segments, usage = translate_segments(
                segments,
                translate_batch=translator,
                max_segments_per_batch=args.batch_size,
                checkpoint=checkpoint,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                progress=progress,
                display=display,
            )
        else:
            translated_segments, usage = translate_segments_openai(
                segments,
                target_language=args.target_language,
                model=model,
                max_segments_per_batch=args.batch_size,
                checkpoint=checkpoint,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                progress=progress,
                display=display,
            )
    except BaseException as exc:
        progress.emit("job", kind="translate", status=job_failure_status(exc), error=str(exc))
        progress.close()
        raise
    finally:
        display.close()
    elapsed = time.perf_counter() - started_at
    result = build_translated_result(
        source=source,
        source_language=source_language,
        target_language=args.target_language,
        translated_segments=translated_segments,
        model=model,
        engine=args.engine,
        token_estimate=estimate,
        usage=usage,
    )
    subtitles, subtitle_summary = prepare_subtitles(
        result, subtitle_options_from_args(args), ("srt", "txt")
    )
    result["subtitles"] = subtitle_summary
    paths = write_translation_outputs(
        result=result,
        subtitles=subtitles,
        output_dir=output_dir,
        stem=stem,
        target_language=args.target_language,
    )

    checkpoint.finish(keep=args.keep_work, prune=prune)
    progress.emit("job", kind="translate", status="complete", outputs=paths)
    progress.close()

    print(f"Elapsed: {elapsed:.2f}s")
    if args.engine == "apple":
        if usage["resumed_batches"]:
            print(f"Resumed batches: {usage['resumed_batches']}/{usage['batches']}")
    else:
        if usage["resumed_batches"]:
            print(
                f"Resumed batches: {usage['resumed_batches']}/{usage['batches']} "
                "(tokens below include them)"
            )
        print(f"Actual input tokens: {usage.get('input_tokens', 0)}")
        print(f"Actual output tokens: {usage.get('output_tokens', 0)}")
        print(f"Actual total tokens: {usage.get('total_tokens', 0)}")
    print(f"Wrote JSON: {paths['json']}")
    print(f"Wrote SRT: {paths['srt']}")
    print(f"Wrote TXT: {paths['txt']}")
    print_subtitle_summary(subtitle_summary)
    return 0


def run_filter(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    formats = parse_formats(args.formats)
    source = load_transcript(input_path)
    source_language = source.get("language") or "unknown"
    stem = transcript_stem(input_path, source_language)
    if "json" in formats and output_dir / f"{stem}.{source_language}.json" == input_path:
        raise ValueError(f"filter would overwrite its input {input_path}; choose another --output")
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered, summary = filter_transcript(
        source,
        FilterOptions(enabled=True, mode=args.filter_mode),
    )
    subtitles, subtitle_summary = prepare_subtitles(
        filtered, subtitle_options_from_args(args), formats
    )
    filtered["subtitles"] = subtitle_summary
    write_outputs(
        result=filtered,
        subtitles=subtitles,
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
    print_subtitle_summary(subtitle_summary)
    return 0


def run_transcription(
    args: argparse.Namespace, audio_path: Path, chunks: list[Chunk], job: dict
) -> dict:
    job["progress"].emit("phase", phase="transcribe", status="started", chunks=len(chunks))
    if chunks:
        return run_chunked_engine(args, audio_path, chunks, job)
    return run_engine(args, audio_path, job.get("duration"))


def run_engine(args: argparse.Namespace, audio_path: Path, duration: float | None = None) -> dict:
    started_at = time.perf_counter()
    language = normalize_language(args.language)
    if duration is None:
        duration = wav_duration(audio_path)
    display = JobProgress(duration, desc="Transcribing", unit="s", clock=True)
    try:
        with whisper_progress(display):
            result = transcribe_audio(
                audio_path,
                model=args.model,
                language=language,
                task=args.task,
                word_timestamps=args.word_timestamps,
                temperature=args.temperature,
                condition_on_previous_text=args.condition_on_previous_text,
                verbose=args.verbose_engine,
                skip_silence=args.skip_silence,
            )
        display.advance_to(display.total)
    finally:
        display.close()
    result["_elapsed_seconds"] = time.perf_counter() - started_at
    return result


def transcription_fingerprint(args: argparse.Namespace, input_path: Path, chunks: list[Chunk]) -> dict:
    """Everything that decides a chunk's result; a change invalidates saved chunks."""
    return {
        "input": file_fingerprint(input_path),
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "temperature": args.temperature,
        "word_timestamps": args.word_timestamps,
        "skip_silence": args.skip_silence,
        "condition_on_previous_text": args.condition_on_previous_text,
        "chunks": [[round(chunk.start, 3), round(chunk.end, 3)] for chunk in chunks],
        "chunk_retries": args.chunk_retries,
        "quality": asdict(QualityOptions()),
    }


def run_chunked_engine(
    args: argparse.Namespace, audio_path: Path, chunks: list[Chunk], job: dict
) -> dict:
    started_at = time.perf_counter()
    progress: ProgressWriter = job["progress"]
    audio = load_wav(audio_path)
    planned_ends = [chunk.end for chunk in chunks]
    chunks = snap_chunks_to_quiet(chunks, audio)
    work_path, prune = work_paths(job["output_dir"], job["stem"], "transcribe", args.work_dir)
    checkpoint, message = Checkpoint.open(
        work_path,
        kind="transcribe",
        fingerprint=transcription_fingerprint(args, job["input_path"], chunks),
        total=len(chunks),
        resume=args.resume,
    )
    progress.emit("phase", phase="transcribe_chunks", status="started", work_dir=str(work_path))
    if message:
        print(message)
    language = normalize_language(args.language)
    results = []
    chunk_reports = []
    # One bar over the whole file's audio time; chunks are an internal detail. A retry
    # re-decodes audio the bar has already counted, so it gets a note instead.
    display = JobProgress(chunks[-1].end, desc="Transcribing", unit="s", clock=True)
    try:
        for chunk in chunks:
            transcribe_chunk(
                args, chunk, audio, language, checkpoint, progress, display, results, chunk_reports,
                total=len(chunks),
            )
            display.advance_to(chunk.end)
    finally:
        display.close()
    resumed = sum(report["resumed"] for report in chunk_reports)
    if resumed:
        print(f"Reused {resumed}/{len(chunks)} chunk(s) saved by an earlier run")

    merged = merge_chunk_results(results, chunks)
    for summary, report, planned_end in zip(
        merged["chunks"], chunk_reports, planned_ends, strict=True
    ):
        summary["elapsed_seconds"] = report["elapsed_seconds"]
        summary["planned_end"] = planned_end
        summary["resumed"] = report["resumed"]
        summary["quality"] = report["quality"]
    merged["_elapsed_seconds"] = time.perf_counter() - started_at
    merged["_checkpoint"] = (checkpoint, prune)
    return merged


FLAG_DESCRIPTIONS = {
    "repeated_text": "the same line repeated",
    "high_compression": "repetitive text",
    "low_coverage": "too little text for the speech in it",
    "no_segments": "no text despite speech",
}


def transcribe_chunk(
    args: argparse.Namespace,
    chunk: Chunk,
    audio,
    language: str | None,
    checkpoint: Checkpoint,
    progress: ProgressWriter,
    display: JobProgress,
    results: list[dict],
    chunk_reports: list[dict],
    *,
    total: int,
) -> None:
    chunk_started_at = time.perf_counter()
    span = f"{format_seconds(chunk.start)}-{format_seconds(chunk.end)}"
    name = f"chunk-{chunk.index:04d}"
    event = {"index": chunk.index, "total": total, "start": chunk.start, "end": chunk.end}
    saved = checkpoint.load(name)
    if saved is not None:
        results.append(saved["result"])
        chunk_reports.append({**saved["report"], "resumed": True})
        progress.emit("chunk", status="resumed", **event)
        return
    progress.emit("chunk", status="started", **event)
    chunk_audio = slice_audio(audio, chunk)
    attempts = []
    temperature = args.temperature
    for retry in range(args.chunk_retries + 1):
        if retry:
            temperature = retry_temperatures(retry)
            problems = " and ".join(FLAG_DESCRIPTIONS.get(flag, flag) for flag in attempts[-1]["flags"])
            display.note(
                f"Re-transcribing {span} at temperature {format_temperature(temperature)}: "
                f"the last pass had {problems}"
            )
        attempt_started_at = time.perf_counter()
        with whisper_progress(display if retry == 0 else None):
            result = transcribe_audio(
                chunk_audio,
                model=args.model,
                language=language,
                task=args.task,
                word_timestamps=args.word_timestamps,
                temperature=temperature,
                condition_on_previous_text=args.condition_on_previous_text,
                verbose=args.verbose_engine,
                skip_silence=args.skip_silence,
            )
        assessment = assess_chunk(result, chunk_audio)
        attempts.append(
            {
                "attempt": retry,
                "temperature": list(temperature)
                if isinstance(temperature, tuple)
                else temperature,
                "elapsed_seconds": time.perf_counter() - attempt_started_at,
                **assessment,
                "_result": result,
            }
        )
        if not assessment["flags"]:
            break

    chosen = min(attempts, key=quality_rank)
    if len(attempts) > 1 and chosen["attempt"] == 0:
        display.note(f"Kept the first pass for {span}; the retry was no better")
    result = offset_result(chosen["_result"], chunk.start)
    for attempt in attempts:
        del attempt["_result"]
    report = {
        "elapsed_seconds": time.perf_counter() - chunk_started_at,
        "quality": {
            "flagged": bool(attempts[0]["flags"]),
            "flags": chosen["flags"],
            "chosen_attempt": chosen["attempt"],
            "attempts": attempts,
        },
    }
    checkpoint.save(name, {"result": result, "report": report})
    results.append(result)
    chunk_reports.append({**report, "resumed": False})
    progress.emit(
        "chunk",
        status="done",
        flags=chosen["flags"],
        elapsed_seconds=round(report["elapsed_seconds"], 3),
        **event,
    )


def normalize_language(language: str) -> str | None:
    normalized = language.strip()
    if normalized.lower() == AUTO_LANGUAGE:
        return None
    return normalized


def resolve_chunk_mode(
    *,
    requested: str | float,
    duration_seconds: float | None,
) -> dict[str, object]:
    if requested == CHUNK_OFF:
        return {"minutes": None, "reason": "explicit off"}
    if duration_seconds is None:
        return {"minutes": None, "reason": "source duration unknown"}
    if requested != CHUNK_AUTO:
        return {"minutes": float(requested), "reason": "explicit chunk length"}
    if duration_seconds > LONG_FORM_THRESHOLD_SECONDS:
        return {
            "minutes": DEFAULT_CHUNK_MINUTES,
            "reason": f"source duration exceeded {LONG_FORM_THRESHOLD_SECONDS}s",
        }
    return {
        "minutes": None,
        "reason": f"source duration did not exceed {LONG_FORM_THRESHOLD_SECONDS}s",
    }


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


def format_temperature(temperature: float | tuple[float, ...]) -> str:
    if isinstance(temperature, tuple):
        return ", ".join(f"{value:g}" for value in temperature)
    return f"{temperature:g}"


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


def prepare_subtitles(
    result: dict,
    options: SubtitleOptions,
    formats: tuple[str, ...],
) -> tuple[dict, dict]:
    """Clean segments for SRT and TXT; JSON output keeps ``result`` untouched."""
    if not {"srt", "txt"} & set(formats):
        return result, {"enabled": False, "reason": "no srt or txt output"}
    return clean_subtitles(result, options)


def write_outputs(
    *,
    result: dict,
    output_dir: Path,
    stem: str,
    language: str,
    formats: tuple[str, ...],
    subtitles: dict | None = None,
) -> None:
    base = f"{stem}.{language}"
    subtitles = subtitles or result
    if "json" in formats:
        write_json(output_dir / f"{base}.json", result)
    if "txt" in formats:
        write_txt(output_dir / f"{base}.txt", subtitles)
    if "srt" in formats:
        write_srt(output_dir / f"{base}.srt", subtitles)


if __name__ == "__main__":
    raise SystemExit(main())
