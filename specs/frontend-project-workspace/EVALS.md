# Harness: Frontend Project Workspace

## Automated Gates

| Gate | Type | Evidence |
| --- | --- | --- |
| API key never re-renders after save | UI/security | DOM assertion and screenshot-free test output |
| SQLite migrations preserve data | integration | migration test with pre/post row counts |
| Project resume works | integration | checkpoint reload and next-work-item assertion |
| Token estimates are deterministic | unit | known text fixture to expected token range |
| Cost simulator is transparent | unit | model price fixture to expected cost |
| Backup excludes secrets | security | archive listing and content scan |
| Backup restores project | smoke | import archive and verify hashes |
| Restore rejects malicious archive | security | zip-slip, traversal, symlink, schema tests |
| Browser never calls OpenRouter directly | UI/security | mocked network assertion |
| Local API rejects hostile requests | integration | Origin, CORS, CSRF, loopback tests |
| Offline path avoids provider import | static/runtime | existing provider boundary pattern |
| Provider calls require consent | unit/integration | mock transport call count |

## Manual Gates

- Verify replacement API key flow does not show the old key.
- Verify backup archive opens on a clean local workspace.
- Verify large DOCX project can be resumed after app restart.
- Verify shared backup mode excludes manuscripts and cost history unless
  explicitly selected.

## Forbidden Fixtures

- Real API keys.
- Commercial manuscript text.
- Real generated audiobook files.
- Raw provider request payloads containing manuscript content.
- Browser-callable provider keys or frontend OpenRouter direct calls.

## Acceptance Criteria

- Every user-visible persistence claim maps to at least one automated or manual
  gate.
- Live provider proof is optional and separate from local readiness.
- Security tests cover key storage, backup content, restore safety, and local API
  boundaries.
