# OpenRouter Audiobook Pipeline Evidence

## Native Gates

```text
py -3 -B -m pytest -p no:cacheprovider -q
47 passed, 1 skipped
```

```text
py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration
passed
```

## FFmpeg Smoke

```text
py -3 -B -m pytest -p no:cacheprovider tests/test_audiobook_e2e_ffmpeg.py -q
1 passed
```

The E2E smoke creates a synthetic DOCX, merges a structured OpenRouter fixture,
marks a generated local WAV, runs `omnivoice.audiobook.mastering_cli`, verifies
the source chunk was not overwritten, probes the final master with `ffprobe`,
and runs `omnivoice.audiobook.qc_cli` with `gate_status=pass`.

## Security and Privacy Gates

```text
secret scan passed
offline provider boundary scan passed
```

Coverage includes untracked feature files. The offline boundary scan covers
`__init__.py`, planner, offline DOCX CLI, workflow CLI, mastering CLI, QC CLI,
and offline audit entrypoints.

## Operational Coverage Added

- `omnivoice-audiobook-workflow merge-openrouter`
- `omnivoice-audiobook-workflow status`
- `omnivoice-audiobook-workflow next`
- `omnivoice-audiobook-workflow mark-generated`
- `omnivoice-audiobook-workflow mark-failed`
- `omnivoice-audiobook-qc`
- `omnivoice-openrouter-audiobook-chunk --preview-only` redacts manuscript text by default.
- `omnivoice-audiobook-workflow next` redacts segment text by default.
- FFmpeg helpers use no-overwrite mode by default and reject source/output path collisions.

## SpecOps Diagnostics

The direct `specops` command is not on PATH in this shell. The Core Business
LittleBull AI wrapper was used instead:

```powershell
& "E:\Empresa IA\codex-master\scripts\invoke-specops-tooling.ps1" -ProjectRoot . validate --root .
& "E:\Empresa IA\codex-master\scripts\invoke-specops-tooling.ps1" -ProjectRoot . eval --root .
& "E:\Empresa IA\codex-master\scripts\invoke-specops-tooling.ps1" -ProjectRoot . report --root .
```

`report` generated `reports/generated/health.json` and
`reports/generated/quality-report.md`; the stable findings are captured here
instead of committing timestamped generated report artifacts.

`specops validate` is blocked by adoption debt unrelated to this slice:
missing base documents such as `specs/constitution.md`, `specs/product-spec.md`,
`docs/security.md`, `CHANGELOG.md`, `governance/license-reuse.md`, and missing
global eval scenario categories.

`specops eval` exited non-zero with `0 passed, 0 failed` because global eval
scenarios are not present.
