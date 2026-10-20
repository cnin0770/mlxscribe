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
uv run mlxscribe transcribe ./movie.mp4 --output ./out
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
  --model mlx-community/whisper-large-v3-turbo \
  --output ./out \
  --formats srt,json,txt
```

Override the source language when you know it:

```bash
uv run mlxscribe transcribe ./movie.mp4 \
  --language ja \
  --output ./out
```

Keep extracted audio for debugging or repeat tests:

```bash
uv run mlxscribe transcribe ./movie.mp4 \
  --output ./out \
  --keep-audio ./work
```

Apply conservative noise filtering:

```bash
uv run mlxscribe transcribe ./movie.mp4 \
  --output ./out \
  --filter-noise
```

The filter is opt-in. It removes empty or near-instant segments, low-confidence silence, and short repetitive leading hallucinations.

Previous-text context defaults to auto:

```text
<= 15 minutes: context on
> 15 minutes: context off
```

Override it when needed:

```bash
uv run mlxscribe transcribe ./movie.mp4 --output ./out --context on
uv run mlxscribe transcribe ./movie.mp4 --output ./out --context off
```

Long files are transcribed in chunks by default:

```text
<= 15 minutes: whole file at once
> 15 minutes: 5-minute chunks
```

Override it when needed:

```bash
uv run mlxscribe transcribe ./movie.mp4 --output ./out --chunk-minutes 10
uv run mlxscribe transcribe ./movie.mp4 --output ./out --chunk-minutes off
```

Chunking costs no speed and did not change accuracy on 45- and 120-minute test files, but it lets a bad stretch be retried without redoing the whole file. Each chunk boundary moves to the quietest point within 15 seconds of its planned position, so chunks split between sentences rather than mid-sentence. Timestamps stay absolute, and the benchmark JSON records each chunk's actual and planned boundaries and its timing.

After each chunk is transcribed, mlxscribe checks it for signs of a bad decode:

- `repeated_text`: the same segment text 8 or more times in a row (a decoding loop)
- `high_compression`: 5 or more segments with a compression ratio of 2.4 or higher (the same limit `--filter-noise` uses)
- `low_coverage`: segments cover less than 30% of a chunk that has audio
- `no_segments`: no segments for a chunk that has audio

Shorter runs of the same text are common in real dialogue ("ん", "はい"), and a single repetitive segment is left to `--filter-noise`.

A flagged chunk is transcribed again at a higher temperature, `(0.2, 0.4)` for the first retry, `(0.4, 0.6)` for the second, and so on. mlxscribe keeps the attempt with the fewest looping segments, preferring any attempt with text over an empty one; on a tie, the earlier attempt wins. `--chunk-retries` sets how many retries a chunk gets (default 1; `0` turns retries off but still reports flags):

```bash
uv run mlxscribe transcribe ./movie.mp4 --output ./out --chunk-retries 2
```

The benchmark JSON's `chunking` section counts `flagged_chunks` and `retried_chunks`, and each chunk's `quality` entry lists every attempt with its temperature, flags and metrics (segment count, longest repeat run, segments in repeat runs, highest compression ratio, coverage, share of audio with sound), plus the `chosen_attempt`.

`--temperature` also accepts a comma-separated list such as `0,0.2,0.4`, which makes mlx-whisper retry each repetitive 30-second window at the next temperature. It is off by default: when every temperature still looks repetitive, mlx-whisper keeps the last attempt, and on the 120-minute test file that produced hundreds of empty segments and ran 40% slower.

Filter an existing transcript without re-running transcription:

```bash
uv run mlxscribe filter ./out/movie.ja.json --output ./out-filtered
```

The benchmark file includes phase timings for probing, audio extraction, transcription, export, and total runtime.

## Benchmarks

```bash
uv run mlxscribe summary out/*.benchmark.json
```

## Translation

Estimate token use without making an API call:

```bash
uv run mlxscribe translate ./out/movie.ja.json --to en --output ./out
uv run mlxscribe translate ./out/movie.ja.json --to zh-Hans --output ./out
```

Run OpenAI translation:

```bash
export OPENAI_API_KEY=...
uv run mlxscribe translate ./out/movie.ja.json \
  --to en \
  --output ./out \
  --engine openai
```

Translation preserves segment timestamps and writes translated JSON, SRT, and TXT files.

Outputs:

```text
out/
  movie.ja.srt
  movie.ja.txt
  movie.ja.json
  movie.benchmark.json
```
