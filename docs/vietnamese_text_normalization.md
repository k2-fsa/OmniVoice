# Vietnamese text normalization

OmniVoice routes `vi`, `vie`, and `vietnamese` through the numeric-only
VietNormalizer adapter when `normalize_text=True`. This backend preserves
non-numeric text, casing, and punctuation, disables transliteration and lexical
rewriting, and uses Unicode Letter/Mark boundaries for short units and currency
markers.

Four-digit years use a cardinal reading in conventional dates; ISO years use
digit-by-digit reading because that is the audited project label, not a claim of
universal preference. Comma decimals are supported only inside money and use
*phẩy* followed by digits. Recognized currencies include Vietnamese đồng,
US/Hong Kong dollars, euro, and yen; other natural regional readings may differ.

## Production and research architecture

The production path is:

```text
API / CLI / batch / Gradio
        -> normalize_for_inference (target only)
        -> VietNormalizerAdapter
        -> VietnameseNormalizer.normalize_numeric
        -> tokenizer/model
```

There is no fallback from VietNormalizer to the custom normalizer. A missing or
incompatible dependency raises a clear error. A recognized per-input runtime
failure preserves the complete raw target and emits a warning; partial output is
never returned.

The earlier custom implementation remains under
`omnivoice/utils/vietnamese_normalization/` for regression research. It has four
explicit boundaries: a
high-recall detector emits non-overlapping candidate spans, a conservative
contextual classifier either selects a reading or abstains, `values.py` parses
the selected span without float conversion, and `verbalizers.py` renders the
typed value deterministically. There is no learned model or global decoder.

The parser/verbalizer layer covers the complete v1 taxonomy: CARDINAL, YEAR,
IDENTIFIER, DECIMAL, FRACTION, DATE, TIME, VERSION, SCORE, RANGE, CURRENCY,
MEASUREMENT, PERCENT, ROMAN, and KEEP. This does not mean every detected form is
automatically rewritten. The current rule classifier only routes cases with
strong structural/context cues; ambiguous decimals, slash forms, scores, ranges,
and codes are preserved until a contextual router is available.

The sole stable inference boundary is
`omnivoice.normalize_for_inference(...)`. Only target text is eligible for
normalization. Reference transcripts remain byte-aligned in meaning with their
reference audio, and voice-design instructions remain in their validated control
vocabulary. `OmniVoice.generate(normalize_text=True)` invokes the boundary once,
before duration estimation and tokenization; the default remains opt-in.

The research backend remains directly inspectable:

```python
from omnivoice.utils.vietnamese_normalization import normalize_with_trace

result = normalize_with_trace("Mã xác nhận là 105.")
print(result.text)
print(result.decisions[0].semiotic_class, result.decisions[0].reason)
```

Its classifier is isolated behind typed spans, so a future learned scorer could
replace it without changing detection or verbalization. Roughly
200 curated seed sentences (and only 30 audio-rated cases) are inadequate for
training or claiming robust generalization from a PhoBERT classifier: they do
not cover enough lexical, domain, or ambiguity variation and lack a held-out
statistical evaluation of useful size.

## Evaluation

Run tests and the four-condition integration benchmark:

```bash
python -m unittest tests.test_vietnamese_normalization -v
.venv/bin/python -m experiments.vietnormalizer_integration.benchmark \
  --dataset experiments/tn_value_gate/data/vi_tn_pilot_v1.jsonl \
  --smoke-cases experiments/vietnormalizer_integration/smoke_cases.jsonl \
  --upstream-python /tmp/omnivoice_vi_benchmark_env/bin/python \
  --improved-python /tmp/vietnormalizer_numeric_env/bin/python \
  --improved-repo /home/pkh257/projects/vietnormalizer \
  --artifacts experiments/vietnormalizer_integration/artifacts
```

The benchmark records case-level outputs for identity, the custom research
normalizer, official VietNormalizer 0.2.3, and the improved numeric-only fork.
The editable fork install is for local development only; production packaging
must wait for review, commit, push, and an immutable dependency revision.

The supplied time oracle `23:59 -> hai mươi ba giờ năm chín phút` was corrected
in the separate regression fixture to *hai mươi ba giờ năm mươi chín phút*.
Digits in a clock minute form a number, not an identifier.

## Known limitations

The pilot data and known regressions were already inspected while designing the
rules, so their scores are development evidence, not held-out generalization.
No local artifact contains completed human audio ratings or reliable old audio
paths. Unsupported input is preserved so downstream TTS does not receive an
invented interpretation.
