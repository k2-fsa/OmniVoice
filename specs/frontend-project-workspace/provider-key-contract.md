# Provider Key Contract

## Contract

The frontend may accept a provider key only as an active user input event. After
save, the key must leave browser state and must never be rendered back to the
user. The persistent project database stores only non-secret metadata.

## Allowed Storage

- OS keyring or equivalent encrypted local credential store.
- Environment variable fallback for runtimes without secure local storage.
- Session-only memory for explicit temporary usage.

## Forbidden Storage

- localStorage
- sessionStorage
- IndexedDB
- service worker cache
- plaintext SQLite
- logs
- backup archives
- provider run records
- frontend bundles or fixtures

## Provider Call Boundary

Browser JavaScript must not call OpenRouter directly. Provider calls go through
a loopback-only local backend that injects credentials server-side.

## Acceptance Criteria

- Key save, replace, remove, and test flows have explicit UI states.
- Tests prove saved keys are absent from DOM, browser storage, SQLite, backup,
  logs, and frontend network calls.
