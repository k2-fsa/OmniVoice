# OpenRouter Audiobook Pipeline Tasks

## Objective

Break OpenRouter Audiobook Pipeline into executable SpecOps tasks.

## Tasks

- [x] Confirm spec and acceptance criteria.
- [x] Implement DOCX chunking for large manuscripts.
- [x] Implement OpenRouter JSON Schema request/response adapter with consent gate.
- [x] Implement segment generation progress/checkpoint helpers.
- [x] Implement FFmpeg concat/remaster helpers.
- [x] Implement workflow CLI for OpenRouter result merge and checkpoint updates.
- [x] Implement QC report CLI for generated plans.
- [x] Add or update tests/evals, including E2E FFmpeg smoke when tooling exists.
- [x] Run native quality gates.
- [x] Record evidence and blockers.

## Acceptance Criteria

- Every task has an owner or next action.
- Validation evidence is available before release.

## Evidence

- `py -3 -B -m pytest -p no:cacheprovider -q` -> `47 passed, 1 skipped`.
- `py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration` -> passed.
- Static secret/provider scan over audiobook code, tests, docs, and specs -> no matches.
- Offline provider boundary scan over offline entrypoints -> passed.
- FFmpeg E2E smoke: synthetic DOCX, structured OpenRouter fixture, checkpoint, local sine WAV, `omnivoice.audiobook.mastering_cli`, `ffprobe`, and `omnivoice.audiobook.qc_cli` -> `1 passed`.
- SpecOps wrapper report generated adoption findings; stable summary recorded in `specs/openrouter-audiobook-evidence.md`.

## Blockers / Adoption Debt

- `specops validate` is blocked by repository-wide SpecOps adoption debt: required base docs and eval scenario categories are missing.
- `specops eval` exits non-zero with `0 passed, 0 failed` because no global eval scenarios exist yet.
- These are not feature regressions from the OpenRouter Audiobook Pipeline slice.
