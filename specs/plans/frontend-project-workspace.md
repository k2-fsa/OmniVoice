# Plan: Frontend Project Workspace

## Phase 1: Contracts and Storage

- Add SQLite schema and migration runner.
- Add path manager for project folders and audio asset paths.
- Add secret-store abstraction with a local fallback policy that does not expose
  secrets in logs or backups.
- Add a loopback-only local API boundary for frontend provider actions. Browser
  code must not call OpenRouter directly.
- Add repository layer for projects, chunks, provider runs, token usage, costs,
  audio assets, QC, checkpoints, and backups.

## Phase 2: Frontend API and Cost Page

- Add route/page for `API & Costs`.
- Add API key configuration panel with save, replace, remove, and test flows.
- Add UI state tests proving saved keys are not rendered after save.
- Add token estimate panel for selected DOCX/project/chunk.
- Add model price table editor and cost simulator.
- Add history table for provider runs and token usage.

## Phase 3: Project Vault UI

- Add project list/detail pages or panels.
- Show source DOCX, chunk status, generation progress, audio assets, QC status,
  total tokens, and total cost.
- Add resume actions: next chunk, continue generation, retry failed segment,
  master audio, run QC.

## Phase 4: Backup and Restore

- Add project export to zip with manifest, SQLite project subset, JSON artifacts,
  audio assets, QC reports, and hashes.
- Add import flow that verifies hashes and reconstructs local paths.
- Exclude secrets and provider credentials from backups.
- Exclude raw provider request payloads and raw provider responses from default
  backups.
- Add restore validation for zip-slip, path traversal, symlinks, schema version,
  and overwrite confirmation.

## Phase 5: Harness and Release

- Add unit tests for SQLite schema, migrations, repositories, token/cost math,
  secret metadata, and backup manifests.
- Add UI tests for key invisibility and cost simulation.
- Add scans that fail on secret literals or provider payloads.
- Add local API security tests for origin, CORS, CSRF, and loopback binding.
- Add smoke tests for backup/restore and resume.

## Acceptance Criteria

- Each phase has an isolated write scope and test plan.
- No phase requires a live provider call for baseline validation.
- Backup and restore are validated before readiness is claimed.
