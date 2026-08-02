# Native Pause Experiment Findings

## Reproducibility

- Source revision: `3d2bd9d07bbe8d16c2439745b0ded450dc41e215`
- Experiment branch: `experiment/latent-pause-control`
- Model revision: `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`
- Aligner revision: `d1d751a5f8271d482d14ca55d9e2deeebbae577f`
- Runtime: Python 3.11.14, Torch 2.8.0+cu128, CUDA 12.8,
  Transformers 5.12.1, tokenizers 0.22.2, RTX 4070 SUPER
- Primary reference: `explaining_sub_11s_24k.wav`, SHA256
  `26b2c81c114eb0d871287a153104a0810ed349fe66ddad3f5474cac3a377b2c0`
- Secondary reference: `explaining_24k.wav`, SHA256
  `6409ff2b4dd0f0a72249e1754c42df47ecd98c64f9a34cb71df2f52f17a25fad`

The system-site-packages virtual environment reports pre-existing dependency
conflicts from the host installation. Editable OmniVoice itself imports from
this fork and reports version `0.2.0`.

## Verified Behavior

- Inline and structured pause parsing, batch validation, duration/speed frame
  planning, room-tone contracts, template placement, CFG template equality,
  diffusion fixed-token checks, mixed mask counts, postprocessing, and API
  compatibility pass automated tests.
- Patched uncontrolled generation exactly matches the pristine seeded 0.2.0
  baseline: shape `8 x 129`, zero differing codec tokens.
- Digital silence encodes to exactly eight codebooks and the requested number
  of 25 Hz frames. Fixed positions remain unchanged through every diffusion
  step.
- Quick 32-step smoke passes for the 0.80-second primary case: raw incremental
  low-energy pause `0.78s`, final incremental pause `0.74s`, WER `0.0`.
- Full 128-step one-pass generation preserves word order and produces WER
  `0.0` for every controlled case. No transient metric flags an abnormal spike.

## One-Pass Acceptance

Full results are tracked in `reports/pause_smoke_results.json`; WAVs and token
tensors are under ignored `outputs/pause_smoke/acceptance/`.

| Case | Requested | Raw incremental | Final incremental | Result |
|---|---:|---:|---:|---|
| Primary 0.40 | 0.40s | 0.50s | 0.53s | Final exceeds tolerance |
| Primary 0.80 | 0.80s | 0.79s | 0.90s | Pass |
| Primary 1.20 | 1.20s | 1.29s | 1.33s | Final exceeds tolerance |
| Secondary 0.80 | 0.80s | 0.67s | 0.63s | Raw and final exceed tolerance |

Tolerance is absolute error at most 120 ms after subtracting the matching
baseline low-energy span. Three of four controlled acceptance cases fail this
strict gate, despite exact fixed-frame counts and intact text.

## Decision

One-pass latent pause control is not sufficient for integration. Conditional
two-pass boundary inpainting was attempted only for the three failed cases. It
inserted fixed pause tokens at the baseline ASR boundary midpoint, remasked five
frames on each side, and preserved every other baseline token.

| Fallback case | Requested | Raw incremental | Final incremental | Result |
|---|---:|---:|---:|---|
| Primary 0.40 | 0.40s | 0.39s | 0.36s | Pass |
| Primary 1.20 | 1.20s | 1.09s | 1.04s | Final exceeds tolerance |
| Secondary 0.80 | 0.80s | 0.78s | 0.84s | Pass |

All fallback cases retain WER `0.0`, exact pause-frame counts, unchanged fixed
tokens, and unchanged baseline tokens outside transition windows. Primary 1.20
still misses the final-output gate by 160 ms. Per experiment rules, no further
tuning or waveform concatenation was attempted.

Final decision: `latent-pause approach failed`. Do not integrate this branch as
production pause control. Keep it as evidence that native fixed-token pauses
work for some durations but do not meet the complete acceptance matrix.

Perceptual click, codec-artifact, and transition-naturalness status:
`manual review pending`. Waveform and ASR metrics cannot approve this gate.

# Single-Reference Narration Control Findings

## Scope

- Experiment revision: `2aa39933b4002d2951df115c9363781d1a9f58d0`
- Branch: `experiment/clone-narration-controls`
- Same model, aligner, reference basename, and reference SHA256 as the pause
  experiment above.
- One `VoiceClonePrompt` was created once and reused for baseline, phrase rate,
  emphasis, intonation, and aside cases. Reference switching was disabled.
- Pronunciation and alias behavior was intentionally excluded.

## Implementation

`NarrationController` first generates a marker-free baseline, aligns requested
words, retains baseline codec tokens outside the selected local window, and
regenerates only the controlled window plus five transition frames. It ranks
multiple candidates using transcript, duration, prominence, or pitch direction.
Output is decoded from model codec tokens; no waveform stitching is used.

Explicit controls support phrase rate, reduced/moderate/strong emphasis,
rising/falling endings, aside delivery, and existing pauses. Optional Markdown-
like shorthand is disabled unless `shorthand=True`.

## Automated Acceptance

The 128-step, three-candidate matrix used one permanent reference. Runtime was
`141.25s` on an RTX 4070 SUPER, with about `2.20 GB` peak allocated GPU memory.
All controlled cases preserved normalized word sequence and fixed codec tokens.

| Control | Baseline | Controlled | Automated result |
|---|---:|---:|---|
| Slow rate `0.85` | 1.64s | 1.92s; 1.92s target | Pass |
| Fast rate `1.20` | 1.64s | 1.38s; 1.36s target | Pass |
| Moderate emphasis | 19 frames | 22 frames; +0.61dB prominence | Pass |
| Strong emphasis | 19 frames, 0.74s | 26 frames, 0.90s | Pass |
| Rising ending | -0.86 semitone slope | 0.00 semitone slope | Pass, subtle |
| Falling ending | -0.86 semitone slope | -2.74 semitone slope | Pass |
| Aside | 1.64s | 1.50s; -1.36dB prominence | Pass |

Full candidate reports, hashes, transcripts, settings, runtime, and GPU memory
are tracked in `reports/narration_control_results.json`. Local A/B WAVs are in
ignored `outputs/narration_controls/acceptance/`.

## Decision

Single-reference local narration control is technically viable. Rate and aside
show strongest objective behavior. Moderate/strong emphasis produce distinct
codec-span and acoustic changes, but emphasis does not always mean louder.
Rising intonation can neutralize a falling baseline yet remain perceptually
subtle. Keep all controls experimental until audio review confirms identity,
naturalness, word stress, transitions, and intended pitch movement.

Listening status: `manual review pending`.

# CMU-Conditioned Narration Findings

## Scope

- Stable legacy commit: `e248cd42dba132849d2f23bb9f29e8fbe86a2ec9`
- CMU implementation: `7561686e3405ff504778d293bdf42d47c925111b`
- Experiment branch: `experiment/cmu-conditioned-controls`
- Dictionary entries verified against CMUdict master and `cmudict.0.7a`.
- Legacy full-codebook inpainting, duration expansion, candidate generation, and
  acoustic ranking remain unchanged.

The controller keeps normal words in cleaned/alignment text and substitutes
validated bracketed ARPABET only in internal generation conditioning. This
avoids treating individual phonemes as expected ASR words.

## Automated Matrix

The 128-step, three-candidate matrix used the original short clone reference.
All controlled and native-bracket cases retained expected ASR words.

| Word | Variant | Frames | Prominence change | Pitch-slope change |
|---|---|---:|---:|---:|
| `unexpected` | Legacy | 13 to 18 | -2.93dB | +1.44 semitones |
| `unexpected` | CMU | 13 to 18 | -5.40dB | +5.76 semitones |
| `nobody` | Legacy | 7 to 9 | -0.25dB | -0.48 semitones |
| `nobody` | CMU primary | 7 to 9 | -1.23dB | -0.65 semitones |
| `nobody` | CMU reduced vowel | 7 to 9 | -0.73dB | +0.24 semitones |

Native bracket generation for plain `genuine` and both official CMU variants
was transcribed as `genuine`. Full reports are tracked in
`reports/cmu_pronunciation_results.json`; local WAVs are under ignored
`outputs/narration_controls/cmu_pronunciation/`.

## Decision

CMU conditioning preserves legacy emphasis freedom and gives explicit phoneme
guidance. Automated metrics cannot determine whether it fixed the audible
mispronunciations, and the CMU variants shifted emphasis away from loudness
toward pitch or duration. Human A/B listening is required before enabling CMU
conditioning by default or combining it with candidate filtering.

Pronunciation, emphasis, and identity status: `manual review pending`.
