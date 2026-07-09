# OpenRouter Structured Processing

This is the online opt-in alternative. It is not part of offline mode.

## Use Case

OpenRouter can structure DOCX-derived chunks into an audiobook plan with model
reasoning for chapter boundaries, dialogue, glossary, speaker hints, and
pronunciation notes.

## Requirements

- Explicit user consent before sending any text.
- API key from environment only.
- User-selected model.
- Structured output or JSON schema support checked for the selected model.
- Chunk preview and estimated cost before full-book processing.
- No full manuscript or provider payload committed to Git.

## Configurable Fields

- `model`
- `temperature`
- `max_tokens`
- `chunk_size`
- `language`
- `genre`
- `rewrite_policy`: `none`, `cleanup_only`, or `structure_only`
- `retention_policy`
- `speaker_detection`
- `pronunciation_mode`

## Stop Conditions

- Missing consent.
- Missing API key.
- Model lacks required structured output support.
- User requests offline guarantee.
- Sensitive manuscript lacks provider-use approval.
