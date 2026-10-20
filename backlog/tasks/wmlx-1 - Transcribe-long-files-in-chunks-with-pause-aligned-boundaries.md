---
id: WMLX-1
title: Transcribe long files in chunks with pause-aligned boundaries
status: Done
assignee: []
created_date: '2026-10-20 04:00'
updated_date: '2026-10-20 04:00'
labels:
  - transcription
dependencies: []
references:
  - commit 82e5cd3
  - commit c9206b7
  - src/mlxscribe/chunking.py
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-file decoding of long videos drifted into loops and gave no way to redo only a bad stretch. Splitting into chunks makes failures local and retryable; fixed-time cuts split sentences, so boundaries need to land in pauses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 --chunk-minutes transcribes sequential chunks with absolute timestamps
- [x] #2 Inner chunk boundaries move to the quietest point within 15s of the planned position
- [x] #3 Each chunk's audio slice is passed to mlx-whisper so the spectrogram is computed per chunk
- [x] #4 Benchmark JSON records each chunk's planned and actual boundaries and timing
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chunked transcription with boundaries snapped to pauses. On sample.16m, 5-minute chunks matched whole-file speed (20.1x vs 19.1x realtime) and output; boundaries landed at 312s/596s/891s between sentences, fixing a sentence split at 300s.
<!-- SECTION:FINAL_SUMMARY:END -->
