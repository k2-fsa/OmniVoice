# OpenRouter Audiobook Pipeline Evidence

## Native Gates

```text
py -3 -B -m pytest -p no:cacheprovider -q
31 passed, 1 skipped
```

```text
py -3 -B -m compileall -q omnivoice\audiobook omnivoice\narration
passed
```

## FFmpeg Smoke

```text
py -3 -B -m omnivoice.audiobook.mastering_cli --input a.wav --input b.wav --output master.wav --tempo 1.0
Wrote remastered audiobook audio: <temp>\master.wav
ffprobe duration: 0.300000
```

## SpecOps Diagnostics

`specops report` created generated reports.

`specops validate` is blocked by adoption debt unrelated to this slice:
missing base documents such as `specs/constitution.md`, `specs/product-spec.md`,
`docs/security.md`, and missing global eval scenario categories.

`specops eval` reported `0 passed, 0 failed` because global eval scenarios are
not present.
