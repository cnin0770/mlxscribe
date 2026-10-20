---
id: WMLX-8
title: Translation glossary for names and recurring terms
status: To Do
assignee: []
created_date: '2026-10-20 04:01'
labels:
  - translation
dependencies: []
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Drama subtitles repeat character and place names that engines translate inconsistently (for example 結夏 rendered differently across lines). A user-supplied glossary keeps them stable across a whole file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 translate accepts a glossary file mapping source terms to target terms
- [ ] #2 OpenAI output follows the glossary; engines that cannot take one say so
- [ ] #3 Tests cover glossary loading and a translated name staying consistent across batches
<!-- AC:END -->
