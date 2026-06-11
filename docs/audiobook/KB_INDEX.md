# OmniVoice Audiobook Knowledge Base

This knowledge base governs the DOCX-to-audiobook workflow for OmniVoice.

## Selected Baseline

Selected implementation baseline: **Alternative A - Offline Deterministic Local**.

Reason: the current OmniVoice distribution is explicitly hardened for local use:
offline Hugging Face/Transformers defaults, no Gradio share, localhost binding, and
no automatic model download. The first production path must preserve that contract.

## Alternatives

| Alternative | Network | Strength | Risk |
| --- | --- | --- | --- |
| A. Offline deterministic local | None | Maximum privacy and repeatability | Less semantic cleanup |
| B. Offline local LLM optional | None after local model is present | Better structure for messy fiction | Must prove model/cache is local |
| C. OpenRouter structured processing | Explicit external API | Better semantic structuring and tool calling | Sends approved chunks to provider |

## Files

- `offline-security-model.md`: how offline mode is proven.
- `docx-ingestion.md`: DOCX extraction and chapter detection rules.
- `audiobook-json-schema.md`: audiobook plan contract.
- `audiobook-standards.md`: pacing, pauses, segment, and QC standards.
- `technical-books-profile.md`: defaults for technical books.
- `fiction-profile.md`: defaults for novels and dialogue-heavy books.
- `qc-mastering.md`: audio handoff and quality gates.
- `openrouter-structured-processing.md`: opt-in external provider path.
