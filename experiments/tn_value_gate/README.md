# Vietnamese text-normalization value gate

This experiment evaluates whether explicit Vietnamese number normalization adds
value before any classifier, ranker, or model fine-tuning is attempted. It is
isolated from the production inference path.

## Dataset and interpretation

The input is the 200-record `pilot.jsonl` stored in Git commit `7053b50`. It is a
**pilot development set**, not an unbiased test set: the current rules were
refined after inspecting some of these examples. The harness preserves the
source data, records its SHA-256, and documents one obvious clock-gold correction
(`time_duration_009`) without rewriting the source. No reliable historical split
or frozen executable was found, so the old 115/160 result cannot be independently
reproduced.

The evaluated systems are the repository's actual current rule, a deliberately
limited integer-span `num2words` adapter, VietNormalizer, and soe-vinorm. The
`num2words` result is a generic verbalization baseline, not a full TN competitor.
Canonical scoring only applies Unicode NFC, whitespace cleanup, and punctuation
spacing cleanup; it does not lowercase or rewrite words.

## Reproduce the text gate

Create the isolated baseline environment (no repository lockfile is changed):

```bash
UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5 uv venv /tmp/omnivoice-tn-value-gate-venv
UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5 uv pip install --python /tmp/omnivoice-tn-value-gate-venv/bin/python \
  num2words==0.5.14 vietnormalizer==0.2.3 soe-vinorm==0.3.2 onnxruntime==1.19.2
```

Materialize the frozen source and run from the repository root:

```bash
git show 7053b50:experiments/vi_number_normalization/data/pilot.jsonl \
  > /tmp/omnivoice_vi_tn_pilot.jsonl
HF_HOME="$HOME/.cache/huggingface" .venv/bin/python -m experiments.tn_value_gate.run_all \
  --dataset /tmp/omnivoice_vi_tn_pilot.jsonl \
  --external-python /tmp/omnivoice-tn-value-gate-venv/bin/python \
  --artifacts experiments/tn_value_gate/artifacts
.venv/bin/python -m experiments.tn_value_gate.audio_pilot.select_cases \
  --outputs experiments/tn_value_gate/artifacts/baseline_outputs.jsonl \
  --best-existing experiments/tn_value_gate/artifacts/best_existing.json \
  --output experiments/tn_value_gate/artifacts/audio_pilot_selection.csv
.venv/bin/python -m experiments.tn_value_gate.audio_pilot.build_manifest \
  --selection experiments/tn_value_gate/artifacts/audio_pilot_selection.csv \
  --artifacts experiments/tn_value_gate/artifacts
```

## Audio gate

Text accuracy cannot answer whether normalization improves OmniVoice audio.
The generated 24-case, 72-stimulus blind manifest compares RAW,
BEST_EXISTING, and GOLD using fixed synthesis settings. Audio generation remains
pending because this machine has no CUDA GPU and no exact reference WAV/transcript.
See `audio_pilot/README.md` for the Colab procedure. Do not treat the text gate or
ASR output as a substitute for blinded listening ratings.

## Tests

```bash
.venv/bin/python -m unittest discover -s experiments/tn_value_gate/tests -v
.venv/bin/python -m compileall -q experiments/tn_value_gate
```
