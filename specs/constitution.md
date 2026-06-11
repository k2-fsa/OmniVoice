# SpecOps Constitution

This repository is primarily an open-source research and developer tool for
OmniVoice text-to-speech. SpecOps artifacts in this branch are an added
governance layer for the DOCX audiobook workflow and must not replace upstream
maintainer policy.

## Principles

- Preserve the existing OmniVoice research and CLI contracts.
- Keep private manuscripts, generated audio, provider payloads, logs, and keys
  out of version control.
- Keep offline DOCX planning separate from any online provider path.
- Require explicit user consent before sending text to OpenRouter or any other
  external provider.
- Do not claim audiobook readiness without test and QC evidence.

## Decision Rules

- Native Python tests and compile gates are authoritative for this branch.
- SpecOps validate and eval are supplemental release gates.
- Live provider checks are optional and approval-gated.
- Release claims must state unverified external dependencies plainly.

## Acceptance Criteria

- Governance files required by SpecOps exist in the repository.
- Safety principles do not conflict with upstream project policy.
- Provider, privacy, and QC boundaries are stated explicitly.
