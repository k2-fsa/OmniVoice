# Vietnamese text-normalization benchmark

This experiment compares unchanged text, the current public OmniVoice API,
OmniVoice's pre-prototype generic `num2words` wrapper, direct `num2words`,
VietNormalizer, soe-vinorm availability, and the frozen seven-class contextual
proof of concept. The current API and contextual adapter intentionally match in
this dirty workspace. It never generates audio.

## Integrity warning

All 200 source records and gold labels were inspected before this benchmark
split was created. The deterministic 181/19 split is therefore a pipeline
rehearsal, not an unseen held-out evaluation. Reuse the schema and splitter on a
new untouched dataset for defensible generalization estimates.

The canonical source is the immutable `pilot.jsonl` blob from Git commit
`7053b50`; its SHA-256 is recorded in `data/derived/dataset_audit.json`. Source
roles are preserved as metadata labels. Domains are absent and remain null.

## Reproduction

```bash
git show 7053b50:experiments/vi_number_normalization/data/pilot.jsonl \
  > /tmp/omnivoice_vi_tn_pilot.jsonl

python -m experiments.vietnamese_text_normalization.split_dataset \
  /tmp/omnivoice_vi_tn_pilot.jsonl \
  --output-dir experiments/vietnamese_text_normalization/data/derived

uv venv /tmp/omnivoice_vi_benchmark_env --python 3.12
uv pip install --python /tmp/omnivoice_vi_benchmark_env/bin/python \
  vietnormalizer==0.2.3 soe-vinorm==0.3.2

python -m experiments.vietnamese_text_normalization.evaluate \
  --canonical experiments/vietnamese_text_normalization/data/derived/canonical.jsonl \
  --splits experiments/vietnamese_text_normalization/data/derived/split_manifest.jsonl \
  --output-dir experiments/vietnamese_text_normalization/results \
  --external-python /tmp/omnivoice_vi_benchmark_env/bin/python

python -m experiments.vietnamese_text_normalization.analyze_failures \
  experiments/vietnamese_text_normalization/results/case_results.jsonl \
  --output-dir experiments/vietnamese_text_normalization/reports
```

Soe-vinorm 0.3.2 is intentionally marked unavailable: constructing its public
normalizer calls `huggingface_hub.snapshot_download` for CRF weights. The task
prohibits model downloads. Its dependency footprint is documented in
`reports/package_availability.json`.

Exact match is case- and punctuation-sensitive. VietNormalizer lowercases full
sentences, so its zero exact-match score must not be interpreted as zero useful
numeric expansions; inspect case-level outputs and failure categories.
