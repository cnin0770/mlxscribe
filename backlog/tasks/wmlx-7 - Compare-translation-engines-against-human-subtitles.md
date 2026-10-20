---
id: WMLX-7
title: Compare translation engines against human subtitles
status: To Do
assignee: []
created_date: '2026-10-20 04:01'
labels:
  - translation
dependencies:
  - WMLX-6
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Engine choice and the default should rest on measured quality, not hand-picked examples. The 120m sample has burned-in bilingual subtitles; OCR already gives 1549 cues with human Chinese lines, and Apple scored 0.19 mean char-bigram F1 against them as a baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Apple, OpenAI and at least one local MLX model are scored against the same human Chinese subtitle reference
- [ ] #2 Results report quality, run time and cost per engine on the 120m sample
- [ ] #3 The scoring script is reproducible from the repo, not only from a scratch directory
<!-- AC:END -->
