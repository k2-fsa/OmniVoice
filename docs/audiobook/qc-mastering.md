# QC and Mastering

QC is mandatory before claiming an audiobook export is ready.

## Checks

- Every expected segment exists.
- Segment files are readable and non-empty.
- Chapter assembly follows JSON order.
- Pauses increase actual duration.
- Joins do not click or truncate words.
- Peaks and loudness are measured when tooling is available.

## Required Report Fields

- `gate_status`: pass, fail, or blocked.
- `duration_seconds`
- `sample_rate_hz`
- `missing_segments`
- `failed_segments`
- `peak_dbfs`
- `loudness_lufs`
- `required_fixes`

## Failure Policy

Do not overwrite source chunks during mastering. Keep generated audio and final
exports separate so regeneration remains possible.
