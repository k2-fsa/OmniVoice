# PRD: DOCX Audiobook Tool

## Objective

Create a local-first tool that turns DOCX manuscripts into structured audiobook
plans for OmniVoice narration generation, with an optional online OpenRouter
structuring path.

## Users

- Authors converting books into narrated audio.
- Operators preparing technical manuals.
- Producers handling novels with chapters and dialogue.

## Core Requirements

- Accept `.docx` input.
- Extract text locally.
- Build a JSON audiobook plan.
- Support technical and fiction profiles.
- Keep offline mode verifiable.
- Reuse existing OmniVoice narration segmentation and TTS assembly contracts.
- Document the OpenRouter path without enabling silent network use.

## Non-Goals

- No automatic paid generation.
- No full distribution mastering claim without QC.
- No manuscript upload in offline mode.
