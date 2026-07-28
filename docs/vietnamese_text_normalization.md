# Vietnamese text normalization

## Runtime contract and default

Vietnamese normalization is opt-in and **off by default** in the Python API,
single/batch CLI, and Gradio. When enabled, only target text is eligible:

```text
target text       -> BamiBERT -> deterministic normalization -> TTS
reference text    -> unchanged (must remain aligned with reference audio)
instruction text  -> unchanged (validated control vocabulary)
```

There is no backend selector and no legacy fallback. If detector loading,
prediction, prediction conversion, or deterministic orchestration fails, the
complete original target is handed onward.

## Candidate-driven BamiBERT pipeline

`omnivoice.text_normalization` is the conservative boundary for externally
supplied BamiBERT spans. It deliberately does not load a model at import time:
callers inject a detector into `normalize_from_ner`, or pass deterministic
`CandidateSpan` objects to `normalize_candidates`.

```text
NER candidates -> trim/verify spans -> label parser -> validation
 -> contextual override -> typed canonical value -> verbalizer
 -> right-to-left replacement
```

Every candidate produces a structured diagnostic (`normalized` or
`preserved`, reason, effective label, canonical value, and replacement).
Invalid offsets, mismatched external surface text, unknown labels, parsing
failures, and losing overlaps leave the source unchanged. Whitespace captured
around the semantic span remains byte-for-byte intact. Overlap resolution is
deterministic: higher confidence, then longer span, then detector order.

`get_bamibert_detector` constructs BamiBERT only on the first enabled
Vietnamese request. A lock makes concurrent Gradio initialization single-copy;
prediction is serialized because tokenizer/pipeline mutation is not documented
as thread-safe. The one-entry cache is keyed by resolved model path and device.
A failed configuration is cached to avoid retrying a large broken load on every
request; a different valid configuration can still replace it, while retrying
the failed configuration requires a restart. The underlying model is put in
`eval()` mode and prediction uses `torch.inference_mode()`.

The default model directory is
`artifacts/models/bamibert_augmented_best`, resolved relative to the repository.
Override it with `OMNIVOICE_BAMIBERT_MODEL` or
`--bamibert-model-path`. The default detector device is CPU; override it with
`OMNIVOICE_BAMIBERT_DEVICE` or `--bamibert-device`.

## Supported labels and conservative formats

The deterministic layer accepts `CARDINAL`, `DECIMAL`, `YEAR`, `DATE`, `TIME`,
`MONEY`, `MEASUREMENT`/`UNIT`, `PERCENT`, `PHONE`, `ID`/`IDENTIFIER`,
`FRACTION`, `SCORE`, and `ORDINAL`. It validates calendar/time bounds, digit
limits, grouping, supported units, required currency markers, fraction
denominators, offsets, external surface text, and overlaps before verbalizing.

Intentionally ambiguous forms remain unchanged. Examples include a lone
`1.234` labeled as a decimal, impossible `31/02/2026`, invalid `25:70`,
unsupported labels, malformed/out-of-range spans, and external surfaces that
do not match the original source. Leading/trailing Unicode whitespace captured
by a span is excluded from replacement so surrounding words cannot concatenate.

## Usage and diagnostics

```bash
omnivoice-infer --language vi --normalize-text \
  --text "Tôi có 25 quyển sách." --output out.wav

omnivoice-infer-batch --normalize-text \
  --test_list cases.jsonl --res_dir results

omnivoice-demo --normalize-text

python scripts/try_text_normalizer.py \
  "Hẹn lúc 08:30 ngày 27/07/2026."
```

The probe prints original text, raw candidates (label, half-open offsets,
surface, score), normalized text, and structured decisions. Use
`--no-diagnostics` to hide the last section. Model/configuration failures
preserve input and return exit status 2.
`scripts/benchmark_text_normalization.py` reports cold loading, warm prediction,
deterministic time, total time, and RSS for representative workloads.

## Step 1 lightweight verification

The default pytest configuration excludes tests marked `integration`, and the
real BamiBERT test also requires the explicit
`OMNIVOICE_RUN_REAL_MODEL_TESTS=1` opt-in. The Step 1 command is therefore safe
for a local environment with limited memory:

```bash
. .venv/bin/activate
python -m pytest -q \
  tests/text_normalization \
  tests/test_normalization_entrypoints.py \
  tests/test_inference_normalization_boundary.py
```

These tests use fake detector factories, predictions, tokenizers, and TTS
outputs. They cover lazy single-copy initialization, warm reuse, cache
isolation, failure recovery through a different valid configuration,
deterministic spans/verbalizers, complete-input fallback, target-only handoff,
entry-point flags, and unchanged reference/instruction fields.

## CPU container

`Dockerfile.cpu` installs CPU PyTorch. `compose.cpu.yaml` mounts BamiBERT at
`/models/bamibert`, persists the Hugging Face cache, and configures the detector
through environment variables. Model files are excluded by `.dockerignore` and
must not be baked into the source image. Its health check imports `omnivoice`
only and therefore cannot trigger model loading.

Real BamiBERT behavior was validated separately on Kaggle. No local real-model
load or inference is part of Step 1, and Step 1 does not claim a real
BamiBERT-to-normalizer run, real OmniVoice synthesis/audio artifact, or Docker
runtime result. Those checks remain intentionally deferred to a suitable
environment.

## Known limitations and regression process

BamiBERT uses its configured tokenizer length; an overlong input that the
pipeline cannot handle fails closed rather than being silently split with
potentially incorrect offsets. Entity confidence and labeling quality remain
model-dependent. Deterministic support is intentionally narrower than the
detector taxonomy.

For a production failure, add the exact source string plus original half-open
candidate offsets to `tests/text_normalization/test_pipeline.py`. Put parser or
verbalizer cases in the corresponding focused test, and use the marked
real-model test only when behavior depends on model prediction.
