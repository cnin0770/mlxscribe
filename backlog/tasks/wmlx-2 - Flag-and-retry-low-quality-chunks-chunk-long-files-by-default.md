---
id: WMLX-2
title: Flag and retry low-quality chunks; chunk long files by default
status: Done
assignee: []
created_date: '2026-10-20 04:00'
updated_date: '2026-10-20 04:00'
labels:
  - transcription
dependencies: []
references:
  - commit ee82e39
  - commit bd1d887
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Loops ("ん" x14, a 30s window repeating "お客様の") and dropouts showed up on long drama audio. Chunking made it possible to detect a bad chunk and redo only that one. Retry thresholds were tuned on the 45m and 120m samples against OCR of their burned-in subtitles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A chunk is flagged for 8+ identical segments in a row, 5+ high-compression segments, low coverage, or no segments despite audio
- [x] #2 A flagged chunk is retried at a higher temperature and the attempt with the fewest looping segments is kept
- [x] #3 --chunk-retries sets retries (default 1, 0 reports flags only)
- [x] #4 --chunk-minutes defaults to auto: 5-minute chunks above 15 minutes, whole file otherwise
- [x] #5 --temperature accepts a fallback list but stays 0.0 by default
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
On the 120m sample, defaults retried 6/25 chunks: loop segments 79 -> 30, high-compression segments 81 -> 15, subtitle cue recall and Japanese similarity unchanged, transcription 297s -> 406s. 45m sample: nothing flagged. A whisper temperature-fallback list was tested and rejected as the default: at T=1.0 it produced 675 empty segments and ran 42% slower.
<!-- SECTION:FINAL_SUMMARY:END -->
