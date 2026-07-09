# SQLite Projects Contract

## Purpose

SQLite is the local source of truth for project continuation, asset indexing,
QC, costs, and backup metadata.

## Required Tables

- projects
- source_documents
- chunks
- audiobook_plans
- provider_runs
- token_usage
- cost_estimates
- audio_assets
- qc_reports
- checkpoints
- backups
- settings
- secret_metadata

## Integrity Rules

- Every child row references a project id.
- Audio assets store path and hash, not audio BLOBs.
- Provider runs store request hashes and response artifact references, not raw
  private payloads by default.
- Checkpoints are appended or versioned so work can be resumed after failure.
- Migrations are versioned and covered by tests.

## Acceptance Criteria

- A project can be closed, reopened, and resumed without data loss.
- SQLite dumps do not contain provider keys or raw manuscript payloads by
  default.
