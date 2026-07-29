# Unicode normalization for inference

Visually identical text can have different Unicode representations. In
particular, non-breaking spaces and invisible format characters can change text
tokenization and severely degrade generated speech. A controlled Vietnamese
test found this with `U+00A0` (non-breaking space) and `U+200B` (zero-width
space).

OmniVoice now applies a small deterministic cleanup to user-supplied target
text and voice-cloning reference transcripts before duration estimation and
tokenization. This is inference preprocessing; it does not change or retrain
the model, tokenizer vocabulary, weights, generation parameters, or audio
decoder.

The cleanup composes text to Unicode NFC, maps `U+00A0`, `U+202F`, and `U+2007`
to ordinary spaces, removes `U+200B`, `U+2060`, and `U+FEFF`, collapses repeated
horizontal whitespace, and trims outer whitespace. Newlines, punctuation,
Vietnamese diacritics, emoji, and other valid content are preserved. Training
data preprocessing is deliberately unchanged.

Run the regression tests without downloading a model:

```bash
python3 -m unittest tests/test_unicode_text_normalization.py
```

To reproduce the audio comparison, first load an `OmniVoice` model in a Python
or Colab session, then pass that existing object to the experiment:

```python
from experiments.unicode_audio_tests.run_experiment import run_experiment

metadata_path = run_experiment(model)
print(metadata_path)
```

The experiment uses auto-voice inference, resets a fixed seed for every case,
and writes WAV files plus `metadata.json` under
`experiments/unicode_audio_tests/results/`. That directory is ignored by Git.
Audio results still depend on the loaded checkpoint and execution backend; the
helper intentionally does not load a checkpoint or require reference audio.
