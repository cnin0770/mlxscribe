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

Expect about 10x realtime on Apple Silicon: an hour of video takes roughly 13 minutes. The defaults favour accurate subtitle timing; add `--skip-silence off` for about twice the speed with less accurate start times (see below).

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

Silence and background noise are skipped by default, so a subtitle starts when its speech starts:

```bash
uv run mlxscribe transcribe ./movie.mp4 --output ./out                      # --skip-silence 2
uv run mlxscribe transcribe ./movie.mp4 --output ./out --skip-silence off   # fast path
```

Without it, Whisper often stamps a line at the very start of the audio (00:00:00) when a video opens with music or noise, and stretches it to the first real words. Chunked runs did the same at every chunk start. `--skip-silence SECONDS` turns on word timestamps and skips silent or noise-only gaps longer than SECONDS, along with text Whisper invents over them. On a 45-minute drama it moved the first line from 0.0s to 10.7s (the burned-in subtitle appears at 11.0s), recovered the opening dialogue that was otherwise missed, and halved the median start-time error (0.57s to 0.35s). It roughly doubles transcription time (131s to 273s, still about 10x realtime). The setting is recorded as `skip_silence` in the benchmark JSON.

It is the default because subtitle timing is what viewers notice. Two cheaper fixes using a speech detector (Silero VAD) were tried on the same file and rejected: moving segment starts onto detected speech barely helped, since most early lines sit inside continuous dialogue with no pause to find; and transcribing only detected speech was faster but lost a whole scene of quiet and phone-filtered dialogue the detector did not register.

Progress is one bar for the whole file, in audio time, however the file is split internally. A chunk that is re-transcribed gets one line above the bar explaining why. The bar is drawn only in a terminal; logs and pipes get plain lines.

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

## Resuming interrupted jobs

Long jobs save their progress as they go. `transcribe` saves each finished chunk (after any retry), and `translate --engine openai` and `--engine apple` save each translated batch, so an interrupted run, a crash, or a failed API request does not throw away finished work. Run the same command again and mlxscribe skips what is already done:

```text
Resuming from out/.mlxscribe-work/movie/transcribe: 3/9 already done.
Transcribing chunks...
  chunk 1/9: 00:00.00-05:00.37 (saved, skipped)
  ...
```

Saved work is reused only when nothing that affects the result changed. For `transcribe` that is the input file (size, modification time and a hash of its first and last MiB), model, language, task, temperature, word timestamps, `--skip-silence`, context mode, the chunk boundaries, `--chunk-retries` and the chunk quality thresholds. For `translate` it is the source segments, source and target language, engine, model (the macOS version for `apple`), batch size and, for `openai`, the prompt. When any of these differ, mlxscribe says which and starts fresh. Resume applies to chunked transcription only; a file short enough to transcribe in one pass has nothing to resume.

An OpenAI request that fails, or returns something that is not the expected JSON, is retried with exponential backoff: `--max-attempts` (default 3) attempts per batch, waiting `--retry-delay` seconds (default 2) and doubling after each failure. If a batch still fails, the command stops with an error, and the batches already paid for stay saved for the next run. Token counts printed after a resumed run include the saved batches.

| Option | Default | Meaning |
| --- | --- | --- |
| `--resume` / `--no-resume` | resume | Reuse matching saved work, or discard it and start fresh |
| `--keep-work` | | Keep the work directory after a successful run |
| `--work-dir DIR` | `<output>/.mlxscribe-work/<input name>` | Where saved work goes |
| `--progress-jsonl FILE` | | Append progress events as JSON lines |

Work lives in `<output>/.mlxscribe-work/<input name>/transcribe/` or `.../translate.<lang>/` and is deleted after a fully successful run. Each holds a `manifest.json` (job kind, settings fingerprint, total units, finished units, and `status`: `running` or `complete`) and one JSON file per finished chunk or batch. The benchmark JSON counts `resumed_chunks` and marks each chunk `resumed: true` or `false`.

`--progress-jsonl` writes one event per line, each with `event` and `time` (Unix seconds): `job` events with `status` `started`, `complete`, `failed` or `interrupted`; `phase` events for audio extraction and transcription; `chunk` events with `index`, `total`, `start`, `end` and `status` `started`, `done` or `resumed`; and `batch` events with `index`, `total` and `status` `started`, `retrying`, `done` or `resumed`.

## Subtitle cleanup

Whisper segments are cut for decoding, not for reading: some run for 15 seconds, some flash for a tenth of a second, and none are wrapped. Before writing SRT and TXT, `transcribe`, `filter` and `translate` turn them into cues that are easy to watch:

- Wrap each cue to at most 2 lines. A line holds at most 42 characters for Latin-script text, or 16 for Chinese, Japanese and Korean, decided per cue by which script most of its characters use. Lines break at sentence ends first, then commas and other clause marks, then spaces, then CJK word boundaries. They never start with closing punctuation or small kana, and katakana words, kanji compounds and words like お金 stay whole where possible.
- Merge short cues (under the minimum duration, or 4 or fewer letters such as "はい") into a neighbour less than 0.5 seconds away, when the result still fits on 2 lines and lasts 7 seconds or less. CJK text joins with a space, or with nothing after punctuation.
- Split cues that do not fit on 2 lines, or that run past 7 seconds, into separately timed pieces. With `--word-timestamps` the pieces take their times from the words. Without them, time is shared out by character count, and a cue split only because it runs long is cut only at a sentence end, comma or space, since the cut's time is a guess. Over-long cues with word timestamps are also trimmed to the first and last word.
- Keep every cue on screen for at least 1 second. Cleanup first extends a short cue into the gap after it, then into the gap before it. Next it tries merging the cue with a neighbour, and last it borrows time from a longer next cue. Overlapping input segments are trimmed, so cues never overlap.

JSON output always keeps the raw segments. The TXT file lists one cleaned cue per line, unwrapped.

Change the limits with these options, on any of the three commands:

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-lines` | 2 | Lines per cue |
| `--max-line-chars` | 42 | Characters per line, Latin-script text |
| `--max-line-chars-cjk` | 16 | Characters per line, Chinese, Japanese or Korean text |
| `--min-cue-seconds` | 1.0 | Minimum time on screen |
| `--max-cue-seconds` | 7.0 | Split cues longer than this |
| `--merge-gap-seconds` | 0.5 | Largest gap a short cue is merged across |
| `--no-subtitle-cleanup` | | Write one unwrapped cue per segment, as before |

```bash
uv run mlxscribe transcribe ./movie.mp4 --language ja --output ./out --max-line-chars-cjk 14
uv run mlxscribe filter ./out/movie.json --output ./out-clean --min-cue-seconds 1.2
```

The changes are recorded under `subtitles` in the benchmark JSON for `transcribe`, and in the output JSON for `filter` and `translate`. The record counts segments split, merged, extended and wrapped, overlaps fixed and cues trimmed to word timestamps. It also lists what cleanup could not fix: `remaining_long_lines`, `remaining_short_cues`, `remaining_long_cues` and `remaining_overlaps`. The usual leftover is `remaining_long_cues`: a cue such as "おかえりなさい" that Whisper stretched over 19 seconds of silence, with no word timestamps to trim it and no pause in the text to split at. The default `--skip-silence` turns on word timestamps, which fixes most of these; they mostly appear with `--skip-silence off`.

## Benchmarks

```bash
uv run mlxscribe summary out/*.benchmark.json
```

## Translation

Whisper only transcribes; translation is a separate step on the transcript JSON. There are three engines:

| Engine | Cost | Runs | Notes |
|---|---|---|---|
| `estimate` (default) | free | locally | Counts tokens for an OpenAI run; translates nothing |
| `apple` | free | on device, offline | macOS Translation framework; translates each line on its own |
| `openai` | paid per token | OpenAI API | Sees the surrounding lines, so it handles names and dialogue better |

### Apple (free, on device)

```bash
uv run mlxscribe translate ./out/movie.ja.json --to zh-Hans --engine apple --output ./out
uv run mlxscribe translate ./out/movie.ja.json --to en --engine apple --output ./out
```

It needs Xcode's command-line tools (`xcode-select --install`) and the language pack for the pair. The first run builds a small Swift helper into `~/Library/Caches/mlxscribe/` (set `MLXSCRIBE_CACHE_DIR` to move it); later runs reuse it. Before translating, mlxscribe checks the pair. If the pack is missing, it stops and tells you to install it in System Settings > General > Language & Region > Translation Languages. `mlxscribe check` reports whether the helper can be built.

Language codes are Apple's: `ja`, `en`, `zh-Hans` (Simplified Chinese), `zh-Hant` (Traditional), `ko`, and so on. A transcript whose language is `zh` is read as `zh-Hans`.

Apple Translation translates each segment without the lines around it. On a 120-minute Japanese drama it ran at about 3.5 segments per second and gave usable drafts, but it guesses the subject when Japanese leaves it out and it can lose slang, wordplay and large numbers. Use it for quick, free drafts and use `openai` for subtitles you will publish. A progress bar counts translated segments and estimates the time left; it moves once per batch (about every 10 seconds with the default `--batch-size 40`).

### OpenAI

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

Both `apple` and `openai` translate in batches of `--batch-size` segments (default 40), retry a failed batch up to `--max-attempts` times, and save each finished batch so an interrupted run resumes (see [Resuming interrupted jobs](#resuming-interrupted-jobs)). The output JSON records the engine under `translation.engine` and, for Apple, the macOS version under `translation.model`; a macOS update starts a fresh translation instead of mixing in batches from the old version.

Translation preserves segment timestamps and writes translated JSON, SRT, and TXT files.

Outputs:

```text
out/
  movie.ja.srt
  movie.ja.txt
  movie.ja.json
  movie.benchmark.json
```
