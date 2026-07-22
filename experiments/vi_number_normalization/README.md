# Vietnamese number-normalization diagnostic

This experiment separates two possible causes of Vietnamese numerical speech
errors:

1. text normalization produces the wrong verbalization; or
2. OmniVoice mispronounces a correct verbalization.

It is a diagnostic evaluation, not a PhoBERT or OmniVoice training workflow.
The manually reviewed source is `data/pilot.jsonl` (200 records) and is never
rewritten by these commands.

## Layout

- `data/pilot.jsonl`: reviewed source records.
- `role_groups.json`: detailed-role to broad diagnostic-group mapping.
- `diagnostic.py`: validation, selection, generation, and evaluation CLI.
- `/tmp/omnivoice_vi_number_diagnostic/`: default generated output, outside
  the repository.

The three systems are:

- **raw:** original `raw_text`, generated with OmniVoice normalization disabled;
- **current:** output from OmniVoice's existing `normalize_text(text, "vi")`,
  then generated with normalization disabled to prevent a second pass;
- **oracle:** `preferred_spoken[0]`, generated with normalization disabled.

## Validate the source

From the repository root:

```bash
python -m experiments.vi_number_normalization.diagnostic validate
```

Validation checks JSON syntax, schema and types, unique IDs, reading lists,
overlap/duplicates, number occurrence, and role coverage. Singleton roles are
reported as warnings. Unknown roles are errors rather than being discarded.

## Select the diagnostic subset

```bash
python -m experiments.vi_number_normalization.diagnostic select \
  --seed 2026 \
  --sample-size 30
```

The default selection is written to
`/tmp/omnivoice_vi_number_diagnostic/selection.json`. It records the seed,
source and mapping hashes, selection policy, and broad-group distribution.
Contrastive same-number families are prioritized before balanced group filling.

## Build manifests without loading TTS

Install the project dependencies, including the existing optional text
normalization support if that is part of the environment being diagnosed:

```bash
pip install -e ".[tn]"
```

Then run:

```bash
python -m experiments.vi_number_normalization.diagnostic generate --dry-run
```

This writes `variants.jsonl` and a blank `evaluation.csv` without loading the
model. The current-system strings come from the repository's real normalizer,
not a reimplementation.

## Generate audio

Use one fixed checkpoint and reference voice for the complete run:

```bash
python -m experiments.vi_number_normalization.diagnostic generate \
  --checkpoint k2-fsa/OmniVoice \
  --reference-audio /path/to/reference.wav \
  --reference-text "Exact transcript of the reference audio." \
  --device cuda:0 \
  --seed 2026 \
  --speed 1.0 \
  --num-step 32
```

The model and reusable voice-clone prompt are initialized once. Every clip is
generated with the same seed and inference settings. Filenames combine a safe
record ID, system, and stable hash. Existing nonempty clips are skipped, and
each success or failure is appended to `generation_results.jsonl`, so a rerun
can resume without losing completed work.

Generated output defaults to `/tmp`. If `--output-dir` points inside this
experiment, use `generated/`, which is Git-ignored.

## Human evaluation and summary

Open `evaluation.csv` and fill only these columns:

- `number_correct`: `yes` or `no`;
- `pronunciation_clear`: `yes` or `no`;
- `naturalness_1_to_5`: integer from 1 to 5;
- `error_type`: use `missing-number`, `repeated-number`, or
  `substituted-number` when applicable, or another consistent label;
- `evaluator_notes`: free text.

Do not infer correctness or naturalness automatically. Incomplete rows are
allowed and excluded from the corresponding denominator.

```bash
python -m experiments.vi_number_normalization.diagnostic summarize \
  --evaluation /tmp/omnivoice_vi_number_diagnostic/evaluation.csv \
  --output /tmp/omnivoice_vi_number_diagnostic/summary.json
```

The summary reports per-system and broad-group rates, naturalness, number-error
counts, fully evaluated raw/current/oracle triplets, cases where oracle fixes a
raw/current failure, and cases where oracle also fails.

## Interpretation

- Current succeeds where raw fails: the existing normalizer likely helps.
- Oracle succeeds where current fails: normalization is the likely bottleneck.
- Oracle also fails: pronunciation/model behavior remains a likely cause.
- Raw and current match: inspect the generated manifest before drawing a
  conclusion; the installed Vietnamese normalization dependencies may have
  left the text unchanged.

Keep the source dataset, role mapping, seed, source hashes, reference voice,
checkpoint, and generation settings fixed when comparing systems.

Out of scope: automatic quality judgments, a complete Vietnamese WFST,
PhoBERT training, OmniVoice fine-tuning, tokenizer changes, and model changes.
