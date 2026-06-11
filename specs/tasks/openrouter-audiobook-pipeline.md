# OpenRouter Audiobook Pipeline Tasks

## Objective

Break OpenRouter Audiobook Pipeline into executable SpecOps tasks.

## Tasks

- [x] Confirm spec and acceptance criteria.
- [x] Implement DOCX chunking for large manuscripts.
- [x] Implement OpenRouter JSON Schema request/response adapter with consent gate.
- [x] Implement segment generation progress/checkpoint helpers.
- [x] Implement FFmpeg concat/remaster helpers.
- [x] Add or update tests/evals.
- [x] Run native quality gates.
- [x] Record evidence and blockers.

## Acceptance Criteria

- Every task has an owner or next action.
- Validation evidence is available before release.

## Evidence

- `py -3 -B -m pytest -p no:cacheprovider -q` -> `31 passed, 1 skipped`.
- `py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration` -> passed.
- Static secret/provider scan over audiobook code, tests, docs, and specs -> no matches.
- FFmpeg smoke: generated two local sine WAVs, concatenated/remastered with `omnivoice.audiobook.mastering_cli`, `ffprobe` duration `0.300000`.

## Blockers / Adoption Debt

- `specops validate` is blocked by repository-wide SpecOps adoption debt: required base docs and eval scenario categories are missing.
- `specops eval` reports `0 passed, 0 failed` because no global eval scenarios exist yet.
- These are not feature regressions from the OpenRouter Audiobook Pipeline slice.
