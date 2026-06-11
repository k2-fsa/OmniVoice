# Telemetry and Privacy

This branch does not add telemetry. The audiobook workflow is local-first.

## OpenRouter

OpenRouter is optional and only used by the dedicated chunk command with
explicit consent. Payloads request provider-side privacy controls:

- `provider.data_collection=deny`
- `provider.zdr=true`

Operators must still treat any online call as a disclosure of the selected
chunk to the provider.
