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

One-pass latent pause control is not sufficient for integration. Decision:
`two-pass required` for failing cases. Boundary inpainting remains conditional
and must preserve all baseline tokens outside five-frame transition windows.

Perceptual click, codec-artifact, and transition-naturalness status:
`manual review pending`. Waveform and ASR metrics cannot approve this gate.
