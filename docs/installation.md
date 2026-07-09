# Installation

Install OmniVoice using the upstream project instructions first. The DOCX
audiobook tooling in PR 184 uses the same Python package and exposes console
scripts through `pyproject.toml`.

## Local Development

```powershell
py -3 -B -m pytest -p no:cacheprovider -q
```

FFmpeg is optional for planning but required for concat, remastering, and audio
QC smoke tests.

## Provider Configuration

OpenRouter is optional. Set `OPENROUTER_API_KEY` and choose a model only when
running a consent-gated online chunk command.
