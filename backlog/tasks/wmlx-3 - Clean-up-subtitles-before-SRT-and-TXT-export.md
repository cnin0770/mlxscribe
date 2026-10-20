---
id: WMLX-3
title: Clean up subtitles before SRT and TXT export
status: Done
assignee: []
created_date: '2026-10-20 04:00'
updated_date: '2026-10-20 04:00'
labels:
  - subtitles
dependencies: []
references:
  - commit 20de93b
  - src/mlxscribe/subtitles.py
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whisper segments are cut for decoding, not reading: some run 15s, some flash for 0.1s, none are wrapped. Subtitles should be pleasant to watch, not just accurate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SRT/TXT lines wrap to at most 2 lines with separate CJK and Latin character limits
- [x] #2 Short adjacent segments merge and over-long ones split into timed pieces
- [x] #3 Cues meet a minimum display time without overlapping the next cue
- [x] #4 JSON keeps raw segments; changes are summarized in the benchmark or output JSON
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleanup runs for transcribe, filter and translate with CLI options for every limit. On sample.16m: 272 segments -> 271 cues, no line over 16 chars, max 2 lines, shortest cue 1.0s, no overlaps. Leftover: cues Whisper stretched over silence; --word-timestamps fixes most.
<!-- SECTION:FINAL_SUMMARY:END -->
