# Product Spec

## Scope

The DOCX audiobook workflow lets an operator transform a local `.docx` book into
an audiobook production plan, optionally structure chunks through OpenRouter,
track segment generation, master audio with FFmpeg, and produce QC reports.

## Users

- Authors preparing fiction or technical books for narration.
- Operators who need resumable chunk-by-chunk audiobook production.
- Developers validating provider boundaries and local audio tooling.

## Core Capabilities

- Local DOCX extraction and chunk preview.
- OpenRouter structured chunk processing only after explicit consent.
- JSON audiobook plan generation and OpenRouter result merge.
- Resumable segment checkpoints with pending, generated, and failed states.
- FFmpeg concat and remastering without overwriting source audio.
- QC report with pass/fail status and required fixes.

## Out of Scope

- Automatic use of private manuscripts in tests.
- Live provider smoke without user approval and a real API key.
- Distribution-platform certification.

## Acceptance Criteria

- The workflow supports local DOCX planning and resumable audiobook operations.
- OpenRouter is optional and consent-gated.
- Technical books and fiction are represented in the documented scope.
