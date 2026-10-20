---
id: WMLX-12
title: Stop tracking backlog/ in git
status: To Do
assignee: []
created_date: '2026-10-20 04:09'
labels:
  - repo
dependencies: []
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Public access to the plans is redundant: ROADMAP.md was already made private, and backlog/ holds the same plans. Not urgent; the user asked to do it with the next delivery. Earlier commits keeping them is fine, so no history rewrite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 backlog/ is listed in .gitignore and removed from the index with git rm --cached, keeping local files
- [ ] #2 The change ships in the next delivery commit, not on its own
<!-- AC:END -->
