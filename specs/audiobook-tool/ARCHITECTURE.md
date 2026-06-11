# Architecture: DOCX Audiobook Tool

## Selected Architecture

Alternative A: Offline deterministic local.

```mermaid
flowchart LR
  A["DOCX manuscript"] --> B["DOCX XML extractor"]
  B --> C["Chapter detector"]
  C --> D["Narration parser"]
  D --> E["Audiobook JSON plan"]
  E --> F["OmniVoice TTS"]
  F --> G["Cache"]
  G --> H["Chapter assembly"]
  H --> I["QC report"]
```

## Modules

- `omnivoice.audiobook.docx`: local DOCX extraction.
- `omnivoice.audiobook.schema`: stable audiobook dataclasses.
- `omnivoice.audiobook.planner`: DOCX-to-plan conversion.
- `omnivoice.audiobook.offline_audit`: offline runtime evidence.
- `omnivoice.audiobook.cli`: command line entrypoint.

## Provider Boundary

OpenRouter belongs behind a separate adapter and must never be called by the
offline planner.
