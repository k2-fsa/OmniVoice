# Frontend Project Workspace Architecture

## Proposed Modules

- `omnivoice/audiobook/storage/schema.py`: SQLite DDL and schema version.
- `omnivoice/audiobook/storage/migrations.py`: migration runner.
- `omnivoice/audiobook/storage/repository.py`: project and asset repositories.
- `omnivoice/audiobook/storage/paths.py`: project folder and asset path layout.
- `omnivoice/audiobook/storage/backups.py`: export/import archive logic.
- `omnivoice/audiobook/storage/secrets.py`: local secret abstraction.
- `omnivoice/audiobook/costing.py`: token estimates and cost simulation.
- Local backend boundary: loopback-only API for frontend provider actions.
- Frontend route/page: `API & Costs`.

## SQLite Tables

- `schema_migrations(version, applied_at)`
- `projects(id, slug, title, author, genre, language, status, created_at, updated_at)`
- `source_documents(id, project_id, original_name, stored_path, sha256, page_estimate, word_count, imported_at)`
- `chunks(id, project_id, source_document_id, chunk_index, text_hash, word_count, estimated_tokens, status)`
- `audiobook_plans(id, project_id, plan_path, plan_hash, version, created_at)`
- `provider_runs(id, project_id, chunk_id, provider, model, consent_at, status, request_hash, response_path, error)`
- `token_usage(id, provider_run_id, estimated_input, estimated_output, actual_input, actual_output, total, source)`
- `cost_estimates(id, provider_run_id, currency, input_cost, output_cost, total_cost, pricing_source, is_actual)`
- `audio_assets(id, project_id, chapter_id, segment_id, role, path, sha256, duration_seconds, sample_rate_hz, channels, status)`
- `qc_reports(id, project_id, audio_asset_id, report_path, gate_status, required_fixes_json, created_at)`
- `checkpoints(id, project_id, checkpoint_path, state_json, created_at)`
- `backups(id, project_id, archive_path, manifest_hash, created_at, verified_at)`
- `settings(key, value_json, updated_at)`
- `secret_metadata(id, provider, fingerprint, created_at, updated_at, last_test_status)`

## Secret Storage Policy

The preferred implementation uses an OS keyring or encrypted local credential
store. If the runtime cannot provide one, the UI must mark the provider as not
securely configurable and require the key through an environment variable for
online calls. Plaintext SQLite storage is not acceptable.

Browser storage is also not acceptable for saved API keys. Do not persist keys
in localStorage, sessionStorage, IndexedDB, service worker cache, SQLite
plaintext, backup archives, logs, or provider run records. The frontend may hold
the key only during the active save request.

## Local Backend Policy

The frontend must not call OpenRouter directly. Online provider calls flow
through a local backend bound to `127.0.0.1` that injects credentials server-side
from the approved secret source. The backend must reject wildcard CORS,
untrusted `Origin`, missing anti-CSRF controls, and non-loopback access.

## Backup Policy

Backups include project SQLite rows, selected artifacts, audio files, QC reports,
and a manifest with hashes. Backups exclude API keys, secret-store payloads,
authorization headers, raw provider request payloads, and raw provider responses
by default. Full manuscript inclusion is an explicit opt-in export mode.

Restore must reject zip-slip, path traversal, symlink overwrite, invalid schema,
and project overwrite without confirmation.

## Acceptance Criteria

- Schema supports full resume and export without provider credentials.
- Secret storage boundary is separate from SQLite project state.
- Audio asset lineage can be reconstructed from database records.
- Frontend-provider calls are mediated by a secure local backend boundary.
