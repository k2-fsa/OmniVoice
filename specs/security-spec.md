# Security Spec

## Secrets

OpenRouter keys must be read from `OPENROUTER_API_KEY` at runtime. Keys, bearer
tokens, provider responses, request payloads, manuscripts, and generated audio
must not be committed.

## Provider Calls

Provider calls require explicit consent. Preview and offline planning must never
send manuscript text to the network. The provider payload requests
`provider.data_collection=deny` and `provider.zdr=true`.

## Logs and Errors

Provider HTTP response bodies are redacted from raised errors by default. CLI
output should identify paths and statuses, not dump secrets.

## Audio Safety

FFmpeg operations must create separate output files and reject output paths that
would overwrite source audio.

## Acceptance Criteria

- No secrets or private manuscripts are committed.
- Provider payloads request data collection denial and zero data retention.
- Audio source overwrite attempts are rejected.
