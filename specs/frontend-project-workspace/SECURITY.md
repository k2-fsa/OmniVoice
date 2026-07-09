# Security: Frontend Project Workspace

## P0 Rules

- Do not persist provider keys in localStorage, sessionStorage, IndexedDB,
  plaintext SQLite, logs, backup archives, fixtures, or committed files.
- Do not call OpenRouter directly from browser JavaScript.
- Do not include provider authorization headers, raw request payloads, raw
  provider responses, or secrets in default backups.
- Do not allow restore archives to write outside the selected project directory.

## Local Backend Boundary

Frontend provider actions must call a local backend bound to `127.0.0.1`. The
backend injects credentials from the approved local secret source and enforces:

- trusted `Origin` checks;
- no wildcard CORS;
- anti-CSRF controls;
- loopback-only binding;
- redacted errors;
- no logging of request bodies containing manuscript text.

## Backup Boundary

Default backups include enough data to restore project state but exclude
secrets, provider payloads, and raw provider responses. Full manuscripts and
cost history are opt-in export categories with manifest disclosure.

## Restore Boundary

Restore must reject zip-slip, path traversal, symlink overwrite, malformed
manifest/schema, hash mismatch, and replacement of an existing project without
confirmation.

## Acceptance Criteria

- The implementation plan has testable controls for browser storage, local API,
  backup content, and restore safety.
- Any fallback that cannot securely store a key must use environment-variable or
  session-only configuration instead.
