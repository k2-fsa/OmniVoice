# Provider Contract: OpenRouter

This contract applies only to the online alternative.

## Inputs

- Approved text chunk.
- Audiobook JSON schema.
- User-selected model config.
- Genre profile.

## Outputs

- Valid audiobook JSON fragment.
- Chapter and segment proposals.
- Pronunciation notes.
- Warnings about uncertain structure.

## Safety Rules

- No provider call without explicit user approval.
- No secrets in code, docs, logs, or tests.
- No silent fallback from offline to online.
- No full manuscript logging.
- Validate model support for structured outputs before use.
