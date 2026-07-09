# Project Memory

## API Project Vault Decision Record

Date: 2026-06-11

The next planned audiobook feature is a local-first API and project vault. It
adds a dedicated frontend page for provider API configuration, SQLite-backed
project preservation, token/cost accounting, audio asset indexing, checkpoints,
and backup/restore.

## Durable Decisions

- API keys may be entered through the frontend, but must not be displayed after
  save.
- SQLite is the project source of truth, but must not store provider keys in
  plaintext.
- Browser code must not call OpenRouter directly. Provider operations must go
  through a loopback-only local backend that injects credentials outside the
  browser.
- Audio files stay on disk; SQLite stores metadata, hashes, paths, and lineage.
- Backups must preserve project state and generated assets, but exclude secrets,
  authorization headers, raw provider requests, and raw provider responses by
  default.
- Restore must defend against zip-slip, path traversal, symlink overwrite,
  malformed schemas, and unconfirmed replacement.
- Token and cost data must distinguish estimates from provider-reported actuals.
- Live OpenRouter testing remains opt-in, approval-gated, and fixture-only.

## Open Implementation Questions

- Which frontend runtime will host the page.
- Which local secret-store mechanism is available on the target platform.
- Whether SQLite should live in the Python backend, desktop shell, or a local
  service boundary.
- Whether backup archives should copy source DOCX files or store references by
  default.

## Acceptance Criteria

- Future implementation work must preserve all durable decisions unless a later
  explicit decision record supersedes them.
- Private manuscripts, generated audio, provider payloads, and secrets must not
  enter version control.
