# Offline Security Model

OmniVoice audiobook mode defaults to local-only execution.

## Required Guarantees

- No automatic model downloads.
- No provider calls in offline mode.
- Gradio must stay bound to `127.0.0.1`.
- `--share` remains blocked.
- Manuscripts, generated audio, JSON plans, caches, and provider payloads must not
  be committed.
- Any online provider path must be explicitly selected by the user.

## Runtime Defaults

The offline runtime must keep:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `GRADIO_ANALYTICS_ENABLED=False`
- `DISABLE_TELEMETRY=1`
- `DO_NOT_TRACK=1`

## Evidence Gates

- Static search for network libraries and provider calls in audiobook code.
- Runtime offline audit through `omnivoice.audiobook.offline_audit`.
- Tests that fail if `network_access_allowed()` stops returning `False`.
- Docker or batch jobs may use `network_mode: none` when running full offline
  conversion and rendering.

## OpenRouter Exception

The OpenRouter path is not offline. It is a separate online mode and must require:

- explicit user consent;
- visible model/config selection;
- chunk-level preview;
- no full manuscript logging;
- no silent fallback from offline mode to provider mode.
