# Evals: DOCX Audiobook Tool

## Required Gates

- Unit: synthetic DOCX extraction.
- Unit: DOCX-to-plan JSON shape.
- Unit: technical vs fiction profile defaults.
- Unit: offline audit passes.
- Static: no OpenRouter/provider call in offline path.
- Unit: OpenRouter payload privacy and consent gate.
- Unit: OpenRouter structured result schema rejects missing or extra fields.
- Unit: workflow checkpoint status, next, generated, and failed transitions.
- Unit: preview and next outputs redact manuscript text by default.
- Unit: FFmpeg command construction rejects source/output overwrite collisions.
- Unit: QC pass/fail for missing, pending, zero-byte, sample rate, channels, peak, and loudness.
- Scale: synthetic 500-paragraph fixture keeps chunks bounded and payloads partial.
- E2E: synthetic DOCX -> structured fixture merge -> checkpoint -> FFmpeg mastering -> ffprobe -> QC CLI when FFmpeg exists.

## Future Gates

- Golden DOCX fixtures for technical and fiction books.
- Chapter assembly duration checks.
- Live OpenRouter smoke with one short public-domain excerpt after explicit approval.
