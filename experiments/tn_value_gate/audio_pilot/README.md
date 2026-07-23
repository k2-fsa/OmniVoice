# Audio value gate (pending)

The manifests contain 24 cases × three conditions = 72 blinded samples. Audio
has not been generated or rated locally. Use one fixed checkpoint, reference
audio, exact reference transcript, seed, GPU, and 16 inference steps.

Colab command:

```bash
python -m experiments.tn_value_gate.audio_pilot.run_colab \
  --manifest experiments/tn_value_gate/artifacts/audio_condition_key.csv \
  --output-dir /content/tn_value_gate \
  --reference-audio /content/reference.wav \
  --reference-text "EXACT transcript supplied by the user" \
  --checkpoint k2-fsa/OmniVoice \
  --device cuda:0 \
  --seed 20260722
```

Upload the repository/worktree, `audio_condition_key.csv`, one reference WAV,
and its exact transcript. Do not guess the transcript. Fill the rating sheet
only after blinded listening; ASR is not a substitute for human correctness.
