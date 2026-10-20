---
id: WMLX-13
title: 'Start subtitles when speech starts, not at 00:00:00'
status: Done
assignee: []
created_date: '2026-10-20 04:59'
updated_date: '2026-10-20 05:22'
labels:
  - transcription
dependencies: []
references:
  - branch apple-translation
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report: the first SRT line always started at 00:00:00 when a video opened with silence or noise. Measured on the 45m sample: Whisper also pinned a segment to the start of every chunk, because chunk boundaries sit in pauses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leading silence or noise does not pull the first subtitle to the start of the file or of a chunk
- [x] #2 A fast path without silence skipping remains available
- [x] #3 The setting is part of the resume fingerprint and recorded in the benchmark
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cheaper alternatives tested on the 45m sample against OCR of the burned-in subtitles (Silero VAD v6.2 via onnxruntime; VAD itself takes 6.5s for 45 min). (1) VAD trim of baseline segments onto detected speech: 138s, lines >5s early 18 -> 15 (skip-silence: 2), first line still 0.0s; the invented opening line overlaps no speech and 90 of the remaining early starts sit inside continuous dialogue with no pause to find. (2) VAD-gated transcription (Whisper hears only detected speech, timestamps mapped back, join-crossing segments split): 117s, best timing (median 0.31s, 1 line >5s early) and least invented text, but lost 25 real lines that baseline and skip-silence both caught, nearly a whole phone-booth scene (2329-2566s) with filtered and whispered voices the VAD did not register. Decision: keep --skip-silence 2 as the default; neither cheaper option is good enough.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
--skip-silence (default 2s; off for the fast path) turns on word timestamps and mlx-whisper hallucination_silence_threshold. 45m sample: first line 0.0s -> 10.7s (subtitle at 11.0s) with the missed opening dialogue recovered, chunk-start-pinned segments 10/10 -> 3/10, median start error 0.57s -> 0.35s, subtitle recall 0.983 -> 0.981, transcription 131s -> 273s (about 10x realtime).
<!-- SECTION:FINAL_SUMMARY:END -->
