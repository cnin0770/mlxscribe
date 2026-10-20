---
id: WMLX-11
title: Native macOS app shell
status: To Do
assignee: []
created_date: '2026-10-20 04:01'
labels:
  - app
dependencies:
  - WMLX-5
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The long-term goal is a Mac app wrapping the proven CLI engine: file picker, job queue, progress, transcript preview, subtitle export and translation settings. It depends on the job manifest and progress events the CLI now emits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An app can pick a file, run transcription and translation through the CLI engine, and show progress from --progress-jsonl
- [ ] #2 Finished subtitles can be previewed and exported from the app
<!-- AC:END -->
