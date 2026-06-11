# Tasks: Frontend Project Workspace

## Spec and Orchestration

- [ ] Confirm frontend stack and routing.
- [ ] Confirm local app runtime boundary for SQLite and secret storage.
- [ ] Confirm local backend boundary for provider calls.
- [ ] Add database schema and migration versioning.
- [ ] Add project folder layout specification.
- [ ] Add backup manifest specification.

## Frontend

- [ ] Add dedicated API/cost page.
- [ ] Add key save flow.
- [ ] Add key replace flow.
- [ ] Add key remove flow.
- [ ] Add non-revealing configured state.
- [ ] Add test-connection flow with explicit consent.
- [ ] Block direct browser OpenRouter calls.
- [ ] Add token estimator.
- [ ] Add cost simulator.
- [ ] Add provider run history.

## Persistence

- [ ] Create `projects` table.
- [ ] Create `source_documents` table.
- [ ] Create `chunks` table.
- [ ] Create `audiobook_plans` table.
- [ ] Create `provider_runs` table.
- [ ] Create `token_usage` table.
- [ ] Create `cost_estimates` table.
- [ ] Create `audio_assets` table.
- [ ] Create `qc_reports` table.
- [ ] Create `checkpoints` table.
- [ ] Create `backups` table.
- [ ] Create `settings` and `secret_metadata` tables.

## Backup and Resume

- [ ] Export full project archive without secrets.
- [ ] Exclude raw provider request and response payloads from default archives.
- [ ] Import project archive with hash verification.
- [ ] Reject zip-slip, traversal, symlink, corrupt, and overwrite-unsafe archives.
- [ ] Resume from latest checkpoint.
- [ ] Preserve failed segment errors for retry.
- [ ] Preserve audio asset lineage from raw to master.

## Quality Gates

- [ ] SQLite migration tests.
- [ ] Repository integration tests.
- [ ] API key invisibility UI test.
- [ ] Secret scan.
- [ ] Offline provider boundary test.
- [ ] Token/cost calculation tests.
- [ ] Backup/restore smoke.
- [ ] Malicious backup restore tests.
- [ ] Local API origin, CORS, CSRF, and loopback tests.
- [ ] Resume workflow smoke.

## Acceptance Criteria

- Tasks cover UI, SQLite, backup, resume, cost, privacy, and harness work.
- No task requires committing private manuscripts or generated audio.
- Live OpenRouter testing remains separate and approval-gated.
