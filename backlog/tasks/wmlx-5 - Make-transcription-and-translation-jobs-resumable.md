---
id: WMLX-5
title: Make transcription and translation jobs resumable
status: Done
assignee: []
created_date: '2026-10-20 04:00'
updated_date: '2026-10-20 04:00'
labels:
  - jobs
dependencies: []
references:
  - commit bbb0c24
  - src/mlxscribe/jobs.py
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An interrupted long run lost all finished work, and a single failed OpenAI request threw away every batch already paid for. The future macOS app also needs a job manifest and progress events from the CLI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chunked transcription saves each finished chunk and a rerun skips them when input and settings match
- [x] #2 Translation saves each finished batch, retries failed batches with backoff, and resumes
- [x] #3 A changed setting or input starts fresh with a message saying what changed
- [x] #4 --progress-jsonl writes structured job, chunk and batch events
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Work lives in <output>/.mlxscribe-work/<stem>/ with a manifest.json and one file per finished unit, removed after success unless --keep-work. Real check: the 45m sample interrupted mid-run resumed and matched an uninterrupted run.
<!-- SECTION:FINAL_SUMMARY:END -->
