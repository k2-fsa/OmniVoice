# CLI Reference

## DOCX Plan

`omnivoice-docx-audiobook-plan` creates a local audiobook JSON plan from a DOCX.

## OpenRouter Chunk

`omnivoice-openrouter-audiobook-chunk` structures one chunk through OpenRouter.
Use `--preview-only` for local preview. Use `--confirm-online-provider` only
when sending that chunk to OpenRouter is intended.

## Workflow

`omnivoice-audiobook-workflow` supports result merge, status, next segment,
mark-generated, and mark-failed operations.

## Mastering

`omnivoice-audiobook-master` concatenates and remasters generated audio with
FFmpeg. It does not overwrite source audio.

## QC

`omnivoice-audiobook-qc` writes a QC report and exits non-zero when required
fixes remain.
