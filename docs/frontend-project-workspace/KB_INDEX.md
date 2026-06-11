# Frontend Project Workspace

The API Project Vault is the planned local-first layer for managing API
configuration, audiobook projects, token/cost accounting, generated audio, and
backup/restore.

## API Key Page

The frontend must provide a dedicated page for provider configuration. After the
key is saved, the UI shows only non-secret metadata: configured status,
provider, fingerprint, last update time, and last test result. The key itself is
never rendered again.

The browser must not call OpenRouter directly. Provider calls go through a local
backend bound to `127.0.0.1`, with trusted-origin checks, anti-CSRF controls, and
no wildcard CORS.

## Local Project Database

SQLite stores project state and indexes every artifact needed to continue work:
DOCX metadata, chunks, plans, provider runs, token usage, cost estimates, audio
assets, QC reports, checkpoints, and backups.

## Audio Project Folder

Audio files remain on disk. A typical project folder should separate source,
chunks, raw audio, mastered audio, QC, and backups. SQLite stores paths and
hashes so the app can resume, verify, export, and restore.

## Token and Cost Controls

The UI shows estimated tokens and estimated costs before online calls. When a
provider returns actual usage, the project history stores actual input/output
tokens and cost values separately from estimates.

## Backup and Restore

Backups are zip archives with a manifest and hashes. They include project state,
selected JSON artifacts, audio assets, and QC reports. They exclude API keys,
authorization headers, raw provider requests, and raw provider responses by
default. Full manuscript inclusion should be an explicit opt-in mode.

Restore must validate hashes and reject path traversal, zip-slip entries,
symlinks, invalid schema, and unconfirmed project replacement.

## Acceptance Criteria

- The user can understand where projects and audio are stored.
- The user can resume, export, save, and restore without exposing secrets.
- Cost and token numbers clearly identify estimated versus actual usage.
