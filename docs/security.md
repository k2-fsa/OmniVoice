# Security

## Manuscripts and Audio

Private manuscripts, generated audio, OpenRouter JSON outputs, checkpoints, and
QC reports are local working artifacts. They must not be committed unless they
are synthetic, small, and explicitly intended as examples.

## Keys

Provider keys must be supplied through environment variables. Do not place keys
in docs, specs, fixtures, command history examples, or logs.

## Network

Offline planning and workflow commands must not load the provider module.
OpenRouter calls require explicit consent and a runtime key.
