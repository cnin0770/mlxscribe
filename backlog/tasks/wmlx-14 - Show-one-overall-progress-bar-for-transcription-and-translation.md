---
id: WMLX-14
title: Show one overall progress bar for transcription and translation
status: Done
assignee: []
created_date: '2026-10-20 04:59'
updated_date: '2026-10-20 04:59'
labels:
  - jobs
dependencies: []
references:
  - branch apple-translation
  - src/mlxscribe/progress_display.py
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report: chunked transcription showed a progress bar per chunk, an internal detail, while Apple translation showed nothing at all, so a long run looked stuck.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transcription shows a single bar over the whole file in audio time, chunked or not
- [x] #2 Per-chunk lines are gone; a retried chunk gets one explanatory note
- [x] #3 Translation with apple or openai shows a segment bar with time remaining
- [x] #4 Bars draw only on a terminal; logs get plain lines
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
mlx-whisper per-call tqdm updates are routed into one JobProgress bar. Checked in a sized pseudo-terminal on sample.16m: one bar, 45 distinct positions across 4 chunks, 00:00/16:43 to 16:43/16:43.
<!-- SECTION:FINAL_SUMMARY:END -->
