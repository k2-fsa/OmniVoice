# Frontend Project Workspace PRD

## Objective

Add a local-first frontend workspace for audiobook production that lets the user
configure provider access through a dedicated page, hide credentials after
saving, persist every project in SQLite, track token and cost usage, and
preserve generated assets for resume, export, and backup.

## Users

- Authors converting technical books or fiction manuscripts into audiobooks.
- Operators generating audiobook chunks manually or continuously.
- Power users who need reliable continuation, extraction, local backups, and
  cost control.

## User Stories

- As a user, I can enter my API key once in a dedicated page and never see it
  displayed again after saving.
- As a user, I can replace or remove the saved key without exposing the old key.
- As a user, I can simulate token and API costs before sending a chunk or a full
  project to an online provider.
- As a user, I can resume any project exactly where I stopped.
- As a user, I can keep every DOCX, chunk, JSON plan, generated audio, QC report,
  cost record, and checkpoint associated with the project.
- As a user, I can export a complete backup and restore it later.

## Functional Requirements

- Provide a frontend page named `API & Costs` or equivalent.
- Persist secret metadata locally while storing the actual key outside plaintext
  SQLite where the runtime supports a local secret store.
- Keep the key invisible in the DOM after save; show only configured status,
  fingerprint, creation date, and replacement/removal controls.
- Never call OpenRouter directly from browser JavaScript. The frontend must call
  a local backend bound to `127.0.0.1`; that backend injects credentials from an
  approved local secret source.
- The local backend must reject untrusted origins, wildcard CORS, missing CSRF
  controls, and access outside loopback.
- Add a SQLite project database for projects, documents, chunks, provider runs,
  token usage, costs, plans, audio assets, QC reports, checkpoints, and backups.
- Store audio files on disk and reference them from SQLite with path, hash,
  duration, sample rate, project id, chapter id, segment id, and asset role.
- Provide backup export/import with manifest and hashes.
- Exclude secrets, authorization headers, raw provider request payloads, and raw
  provider responses from backup by default.
- Treat full manuscript inclusion in backup as explicit opt-in with a clear
  manifest inventory.
- Support both manual one-chunk generation and continuous generation with
  checkpoint persistence after each unit of work.

## Non-Functional Requirements

- Local-first by default.
- No live provider call without explicit consent.
- No private manuscripts, audio, request payloads, or keys in Git fixtures.
- SQLite migrations must be additive and reversible by backup.
- Restore must defend against zip-slip, path traversal, symlink overwrite,
  invalid schema, and replacement of an existing project without confirmation.
- Cost values must identify whether they are estimates or provider-reported
  actuals.
- Cost values are project-sensitive metadata and must be optional in shared
  backups.
- UI must include loading, empty, success, error, and destructive-confirmation
  states.

## Out of Scope

- Cloud synchronization.
- Multi-user auth.
- Payment processing.
- Upstream release publication.
- Live OpenRouter smoke without approval and a runtime key.

## Acceptance Criteria

- A dedicated frontend page exists in the implementation plan with key save,
  replace, remove, and test-connection flows.
- Saved API keys are never displayed after save and are excluded from backups.
- No implementation stores API keys in localStorage, IndexedDB, plaintext
  SQLite, logs, backup archives, or frontend application state beyond the active
  save request.
- Browser code never calls OpenRouter directly with a provider key.
- SQLite schema covers all project continuation and export needs.
- Token and cost simulation is defined before online generation.
- Backup and restore preserve project state and audio references.
- Harness/evals include API-key privacy, SQLite resume, cost simulation, backup,
  and offline-provider-boundary checks.
