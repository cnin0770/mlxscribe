---
id: WMLX-6
title: Add on-device Apple Translation engine
status: In Progress
assignee: []
created_date: '2026-10-20 04:01'
updated_date: '2026-10-20 04:02'
labels:
  - translation
dependencies: []
references:
  - branch apple-translation
  - src/mlxscribe/apple_translation.py
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
OpenAI translation costs money per run, which discourages trying things. macOS ships a free, offline Translation framework that works from a command-line helper, so translations can be tested whenever needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 translate --engine apple translates with the macOS Translation framework, no API key
- [x] #2 A missing or unsupported language pack stops the run before translating, with install instructions
- [x] #3 Apple batches get the same retry, checkpoint and resume behaviour as OpenAI
- [ ] #4 The user has tried it on the apple-translation branch and approved merging into main
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implemented on branch apple-translation (one commit on top of main). Remaining: user trial on the branch, then fast-forward into main and push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified: 102 tests pass plus an opt-in test that compiles and runs the real Swift helper (MLXSCRIBE_APPLE_TESTS=1). 120m transcript (1721 segments, ja -> zh-Hans) took 493s, about 3.5 segments/s. Ctrl-C after 4/44 batches resumed and gave output identical to an uninterrupted run. Baseline: mean char-bigram F1 0.19 against OCR of the human Chinese subtitles. Quality is literal: no context between lines, guessed subjects, lost slang and wordplay, one wrong large number in English.
<!-- SECTION:NOTES:END -->
