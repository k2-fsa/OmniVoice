# Quality Gates

## Required Local Gates

```powershell
py -3 -B -m pytest -p no:cacheprovider -q
py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration
```

## Required Static Gates

- Scan for OpenRouter key prefixes, environment assignments, and real bearer
  tokens without committing those literal secret values.
- Check offline entrypoints for provider or network imports.
- Confirm private output patterns are ignored.

## Required Audio Gates

- If FFmpeg is present, generate synthetic WAV fixtures only.
- Run concat/remaster smoke.
- Use `ffprobe` or QC CLI to confirm readable audio metadata.

## Optional Gates

- Live OpenRouter smoke with explicit user approval, a real key, and only a
  public-domain fixture.

## Acceptance Criteria

- Native tests and compile gates pass before delivery.
- Static privacy and provider boundary checks pass.
- FFmpeg smoke is run when FFmpeg is available.
