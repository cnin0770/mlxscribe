# mlxscribe

`mlxscribe` is a local-first transcription CLI for macOS/Apple Silicon. The POC extracts audio with `ffmpeg`, transcribes with `mlx-whisper`, and exports text, JSON, and SRT files.

## Requirements

- macOS on Apple Silicon
- `uv`
- `ffmpeg` and `ffprobe`

## Quick start

```bash
uv sync
uv run mlxscribe check
uv run mlxscribe transcribe ./movie.mp4 --language ja --output ./out
```

The default model is `mlx-community/whisper-large-v3-turbo`.

Model downloads are cached in the project directory by default:

```text
.hf-cache/
```

Set `HF_HOME` before running the CLI if you want to use a different Hugging Face cache.

## Example

```bash
uv run mlxscribe transcribe ./movie.mp4 \
  --language ja \
  --model mlx-community/whisper-large-v3-turbo \
  --output ./out \
  --formats srt,json,txt
```

Keep extracted audio for debugging or repeat tests:

```bash
uv run mlxscribe transcribe ./movie.mp4 \
  --language ja \
  --output ./out \
  --keep-audio ./work
```

The benchmark file includes phase timings for probing, audio extraction, transcription, export, and total runtime.

Outputs:

```text
out/
  movie.ja.srt
  movie.ja.txt
  movie.ja.json
  movie.benchmark.json
```
