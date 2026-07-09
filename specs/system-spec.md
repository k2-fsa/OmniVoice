# System Spec

## Boundaries

The audiobook system is implemented under `omnivoice/audiobook`. Offline
entrypoints must not import or load the OpenRouter provider module. Online
provider functionality is isolated behind `omnivoice-openrouter-audiobook-chunk`.

## Data Flow

1. DOCX is parsed locally.
2. Text is chunked with deterministic limits suitable for books up to 500 pages.
3. Preview mode writes local chunk metadata without provider calls.
4. With explicit consent, one chunk may be sent to OpenRouter for structured JSON.
5. Structured JSON files are merged into an audiobook plan.
6. Operators mark segment audio as generated or failed.
7. FFmpeg creates separate concat/remaster outputs.
8. QC inspects generated audio and returns a gate status.

## Required Controls

- API keys come from environment variables only.
- Provider errors redact response bodies.
- Generated audio and private JSON artifacts are ignored by default.
- Source audio cannot be overwritten by concat or remaster operations.

## Acceptance Criteria

- Offline entrypoints do not import or load the OpenRouter module.
- Online provider calls require explicit consent and a runtime API key.
- QC and mastering commands fail closed on missing or unsafe inputs.
