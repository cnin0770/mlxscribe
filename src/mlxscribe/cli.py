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
from mlxscribe.engine import transcribe_audio
from mlxscribe.exporters.json_exporter import write_json
from mlxscribe.exporters.srt import write_srt
from mlxscribe.exporters.txt import write_txt
from mlxscribe.formats import parse_formats
from mlxscribe.media import extract_audio, probe_duration

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FORMATS = ("srt", "json", "txt")
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
        default="ja",
        help="Source language code passed to Whisper. Defaults to ja.",
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
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use prior transcript text as context for later windows.",
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
    transcribe_parser.set_defaults(command="transcribe")

    return parser


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
        result = run_engine(args, audio_path)
        phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")
    else:
        with tempfile.TemporaryDirectory(prefix="mlxscribe-") as temp_dir:
            audio_path = Path(temp_dir) / f"{stem}.wav"
            phase_started_at = time.perf_counter()
            extract_audio(input_path, audio_path)
            phase_timings["extract_audio_seconds"] = time.perf_counter() - phase_started_at
            result = run_engine(args, audio_path)
            phase_timings["transcribe_seconds"] = result.pop("_elapsed_seconds")

    phase_started_at = time.perf_counter()
    write_outputs(
        result=result,
        output_dir=output_dir,
        stem=stem,
        language=args.language,
        formats=formats,
    )
    phase_timings["export_seconds"] = time.perf_counter() - phase_started_at
    elapsed = time.perf_counter() - started_at
    phase_timings["total_seconds"] = elapsed

    benchmark = {
        "input": str(input_path),
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "formats": formats,
        "word_timestamps": args.word_timestamps,
        "temperature": args.temperature,
        "condition_on_previous_text": args.condition_on_previous_text,
        "hf_home": str(get_hf_home()),
        "kept_audio": str(kept_audio_path) if kept_audio_path else None,
        "source_duration_seconds": source_duration,
        "elapsed_seconds": elapsed,
        "realtime_factor": (source_duration / elapsed) if source_duration else None,
        "phase_timings": phase_timings,
    }
    benchmark_path = output_dir / f"{stem}.benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {output_dir}")
    if source_duration:
        print(f"Elapsed: {elapsed:.2f}s ({source_duration / elapsed:.2f}x realtime)")
    else:
        print(f"Elapsed: {elapsed:.2f}s")
    print_phase_timings(phase_timings)
    return 0


def run_engine(args: argparse.Namespace, audio_path: Path) -> dict:
    print("Transcribing...")
    started_at = time.perf_counter()
    result = transcribe_audio(
        audio_path,
        model=args.model,
        language=args.language,
        task=args.task,
        word_timestamps=args.word_timestamps,
        temperature=args.temperature,
        condition_on_previous_text=args.condition_on_previous_text,
        verbose=args.verbose_engine,
    )
    result["_elapsed_seconds"] = time.perf_counter() - started_at
    return result


def print_phase_timings(phase_timings: dict[str, float]) -> None:
    print("Phase timings:")
    for name, seconds in phase_timings.items():
        label = name.removesuffix("_seconds").replace("_", " ")
        print(f"  {label}: {seconds:.2f}s")


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
