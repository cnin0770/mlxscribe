---
id: WMLX-9
title: Per-file transcript quality report
status: To Do
assignee: []
created_date: '2026-10-20 04:01'
labels:
  - subtitles
dependencies: []
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Before publishing, it is hard to know where a transcript is weak: flagged chunks, dropped segments and suspicious cues are scattered across the benchmark JSON. One summary should say where to look.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A report lists flagged and retried chunks, segments dropped by --filter-noise, and cues cleanup could not fix, with timestamps
- [ ] #2 The report is written next to the outputs as readable text and as JSON
- [ ] #3 Tests cover a report built from a recorded benchmark
<!-- AC:END -->
