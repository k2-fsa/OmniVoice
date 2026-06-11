# Evals: DOCX Audiobook Tool

## Required Gates

- Unit: synthetic DOCX extraction.
- Unit: DOCX-to-plan JSON shape.
- Unit: technical vs fiction profile defaults.
- Unit: offline audit passes.
- Static: no OpenRouter/provider call in offline path.

## Future Gates

- Golden DOCX fixtures for technical and fiction books.
- Chapter assembly duration checks.
- QC report fixture.
- Provider adapter schema validation with mocked OpenRouter responses.
- End-to-end local smoke with one short public-domain excerpt.
