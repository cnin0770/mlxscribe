# mlxscribe Roadmap

## 1. Transcription Stability

- Chunked transcription
- Per-chunk retry behavior
- Loop and hallucination reporting
- Optional VAD or speech detection
- Resume partial jobs (done: chunk and translation-batch checkpoints)
- Full-file versus chunked benchmark comparison

Goal: long videos should be boringly reliable.

## 2. Transcript Quality Tools

- Subtitle line wrapping
- Maximum characters per subtitle
- Merge or split short segments
- Remove or flag suspicious segments
- Confidence and quality report
- Searchable JSON audit output

Goal: produce subtitles that are not only accurate, but pleasant to watch.

## 3. Translation Pipeline

- Estimate token cost
- OpenAI translation engine
- Translated SRT, TXT, and JSON output
- Preserve timestamps
- Glossary or name dictionary
- Batch retry
- Compare English and Chinese outputs
- Consider Apple Translation in the future native app

Goal: reliable `transcribe -> translate -> subtitle` flow.

## 4. Job System

- Job manifest
- Input, output, and work directories
- Phase progress
- Per-job logs
- Resumable chunks
- Structured JSONL progress events

Goal: make the CLI a clean engine for a future macOS app shell.

## 5. Model And Runtime Management

- Model download, list, and check commands
- Model cache location management
- Model metadata in benchmarks
- Model variant comparison
- Warm-up timing
- Preset profiles: fast, balanced, quality, longform

Goal: make model use reproducible and understandable.

## 6. Native macOS Shell

- File picker
- Job queue
- Progress UI
- Transcript preview
- Subtitle export
- Translation settings
- Model and cache settings
- Possible Apple Translation integration

Goal: wrap a proven local engine in a clean Mac app.

## Recommended Near-Term Sequence

1. Chunked transcription
2. Subtitle cleanup and wrapping
3. OpenAI translation execution polish
