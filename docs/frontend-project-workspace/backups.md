# Backups

Backups preserve project state, generated assets, and enough metadata to restore
work on another local workspace.

## Default Exclusions

- Provider keys.
- Authorization headers.
- Raw provider requests.
- Raw provider responses.
- Secret-store payloads.

## Restore Safety

Restore validates hashes and rejects path traversal, zip-slip, symlinks,
malformed schema, and unconfirmed overwrite.
