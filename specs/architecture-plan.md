# Architecture Plan

## Modules

- `docx.py`: extracts local DOCX document structure.
- `chunking.py`: creates semantic audiobook chunks.
- `openrouter.py`: contains the isolated online provider client.
- `openrouter_cli.py`: exposes consent-gated online chunk structuring.
- `planner.py`: creates and merges audiobook JSON plans.
- `generation.py`: manages resumable segment checkpoints.
- `workflow_cli.py`: operates merge/status/next/mark workflows.
- `mastering.py`: wraps FFmpeg/ffprobe operations.
- `mastering_cli.py`: exposes concat and remaster commands.
- `qc.py`: validates generated segment audio.
- `qc_cli.py`: writes QC report JSON and fails closed on blocking issues.

## Patterns

- Keep provider code isolated from offline imports.
- Prefer explicit command flags over implicit online behavior.
- Use structured JSON contracts for plans and provider outputs.
- Keep audio transformations non-destructive.

## Acceptance Criteria

- Module ownership remains small and explicit.
- CLI entrypoints map to their documented module responsibilities.
- Provider, workflow, mastering, and QC concerns stay separated.
