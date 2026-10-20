---
id: WMLX-4
title: Keep the language code in output filenames
status: Done
assignee: []
created_date: '2026-10-20 04:00'
updated_date: '2026-10-20 04:00'
labels:
  - transcription
dependencies: []
references:
  - commit f8e4e27
  - commit b52a22a
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Path.with_suffix replaced the ".ja" in "movie.ja", so transcripts were written as movie.json instead of the documented movie.ja.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcribe, filter and translate write <stem>.<lang>.{json,srt,txt}
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
File names are built by appending the extension. Merged together with subtitle cleanup in b52a22a; subtitle tests updated to expect movie.ja.*.
<!-- SECTION:FINAL_SUMMARY:END -->
