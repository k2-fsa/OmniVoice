# Release Process

## Branch Readiness

Before asking for merge:

```powershell
py -3 -B -m pytest -p no:cacheprovider -q
py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration
```

Also run secret scans, provider boundary checks, FFmpeg smoke when available,
and SpecOps `validate`, `eval`, and `report`.

## Release Authority

This branch can document readiness evidence but does not grant authority to
publish an upstream release.
