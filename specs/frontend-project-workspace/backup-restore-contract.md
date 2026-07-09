# Backup Restore Contract

## Backup Contents

Default project backups may include:

- selected project SQLite rows;
- project manifest;
- DOCX metadata and optionally source DOCX;
- structured JSON artifacts;
- audio assets;
- QC reports;
- token and cost summaries when selected;
- hashes for every included file.

Default project backups must exclude:

- provider keys;
- authorization headers;
- raw provider request payloads;
- raw provider responses;
- secret-store payloads.

## Restore Rules

- Reject path traversal and zip-slip entries.
- Reject symlinks and unsafe file modes.
- Reject manifest hash mismatch.
- Reject malformed schema.
- Require confirmation before replacing an existing project.
- Restore into a controlled project directory only.

## Acceptance Criteria

- Export/import round-trip preserves project state and audio asset records.
- Malicious archives fail closed.
- Shared backup mode excludes secrets and private payloads.
