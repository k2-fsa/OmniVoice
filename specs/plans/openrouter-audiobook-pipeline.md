# OpenRouter Audiobook Pipeline Plan

## Objective

Implement the smallest useful vertical slice for OpenRouter Audiobook Pipeline.

## Architecture

- Use shared core logic when possible.
- Keep storage local-first.
- Avoid paid or secret-bearing services.
- Keep OpenRouter behind explicit consent and environment-only credentials.
- Use JSON Schema structured output and local validation before plan merge.
- Keep FFmpeg as an optional local post-processing boundary with clear errors.

## Tasks

1. Update spec and acceptance criteria.
2. Implement scoped behavior.
3. Add tests or eval scenarios.
4. Run validation gates.
5. Update release evidence.

## Acceptance Criteria

- Scope is implemented without unsafe overwrites.
- Tests and validation gates pass.
