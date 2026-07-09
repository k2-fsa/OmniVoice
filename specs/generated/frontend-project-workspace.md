# Generated Spec: Frontend Project Workspace

## Summary

The Frontend Project Workspace feature adds a dedicated frontend surface and local
persistence layer for managing OpenRouter credentials, audiobook projects,
token/cost accounting, generated audio assets, and backups. It builds on the
existing DOCX audiobook pipeline without changing the rule that online provider
calls require explicit consent.

## Key Decisions

- Store the actual API key through a local secret abstraction, not plaintext
  SQLite.
- Store only secret metadata in SQLite: provider, configured status,
  non-secret fingerprint, created/updated timestamps, and last test result.
- Route frontend provider actions through a loopback-only local backend; never
  expose provider keys to browser-side OpenRouter requests.
- Use SQLite as the source of truth for projects and state.
- Keep audio files on disk in project folders; SQLite stores metadata and paths.
- Treat backups as portable project archives that exclude secrets by default.
- Token/cost records distinguish estimated values from provider-reported actuals.

## Data Ownership

- SQLite owns project metadata, state, costs, and asset index records.
- Filesystem owns DOCX copies, generated JSON files, raw audio, mastered audio,
  QC reports, and backup archives.
- Secret store owns API keys.

## Risk Controls

- API key must never be rendered after save.
- API key must not be written to logs, SQLite plaintext, backups, test fixtures,
  or Git.
- API key must not be stored in localStorage, IndexedDB, sessionStorage, or
  browser-accessible long-lived state.
- Local backend must enforce trusted origin, anti-CSRF, loopback binding, and no
  wildcard CORS before provider operations.
- Offline entrypoints must not load provider clients.
- Backup restore must verify manifest hashes and reject zip-slip/path traversal,
  symlink overwrite, malformed schemas, and unconfirmed replacement before
  marking a project restored.
- Continuous generation must checkpoint after each chunk or segment.

## Acceptance Criteria

- The implementation plan defines frontend, persistence, backup, token/cost, and
  privacy slices.
- All sensitive storage boundaries are explicit.
- The feature can be implemented incrementally without breaking PR 184 pipeline
  behavior.
