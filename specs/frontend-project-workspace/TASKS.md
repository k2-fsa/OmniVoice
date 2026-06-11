# Frontend Project Workspace Task Registry

## Workstreams

- API key UI and local backend boundary.
- SQLite project database and migrations.
- Token and cost simulator.
- Project/audio asset vault.
- Backup and restore.
- Harness, evals, and security scans.

## Sequencing

1. Implement storage and migrations before frontend persistence.
2. Implement secret metadata and local backend boundary before provider testing.
3. Implement token/cost estimation before continuous generation.
4. Implement backup manifest before restore.
5. Implement restore safety before declaring backup readiness.

## Acceptance Criteria

- Workstreams are independently testable.
- No task requires private manuscripts or real provider keys in fixtures.
- Each P0 security rule maps to at least one gate.
