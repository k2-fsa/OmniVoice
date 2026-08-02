# Native Pause and Single-Reference Narration Controls

## Purpose

This document records the OmniVoice experiments implemented on top of version
`0.2.0`. The work stays inside the OmniVoice package. It does not depend on the
GenTest viewer or any website UI.

The main goal was to control pauses and local delivery while keeping one voice
clone reference. Changing reference audio was explicitly avoided because even
references made from the same identity produced small identity and style shifts.

The branch contains three related systems:

1. Native fixed-token pauses in `OmniVoice.generate()`.
2. Single-reference narration controls through `NarrationController`.
3. Generation traces and a standalone native diagnostic capture script.

Pronunciation-safe protected-codebook inpainting was also tested on a separate
branch. It is documented here as a rejected experiment and is not included in
the current implementation.

## Revisions and Runtime

- Upstream source: `3d2bd9d07bbe8d16c2439745b0ded450dc41e215`
  (`0.2.0`).
- Model snapshot: `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
- Alignment snapshot: `d1d751a5f8271d482d14ca55d9e2deeebbae577f`.
- Main experiment revisions:
  - `dc21d79`: native pause token controls.
  - `7ed42cb`: conditional boundary-inpainting fallback.
  - `2aa3993`: single-reference narration controls.
  - `7561686`: CMU-conditioned narration controls.
  - `8e6dcb9`: generation traces.
  - `7a55ebd`: native generation diagnostic capture.
- Acceptance runtime: Python 3.11.14, Torch 2.8.0+cu128, CUDA 12.8,
  RTX 4070 SUPER.

## Native Pause Controls

### Public API

Inline pause markers are removed before tokenization:

```python
audio = model.generate(
    text="The first result passed.<pause:0.80> The second exposed the bug.",
    ref_audio="ref.wav",
    ref_text="Exact reference transcript.",
)
```

Structured plans use offsets in cleaned text:

```python
from omnivoice import PausePlan, PauseSpec

text = "The first result passed. The second exposed the bug."
audio = model.generate(
    text=text,
    pause_plan=PausePlan((PauseSpec(after_char=24, seconds=0.8),)),
    ref_audio="ref.wav",
    ref_text="Exact reference transcript.",
)
```

`pause_audio` accepts either a path or `(waveform, sample_rate)`. When omitted,
the audio tokenizer encodes exact digital silence. When supplied, the source is
converted to mono 24 kHz and center-cropped for every requested pause.

### Parsing and Validation

- Marker parsing happens before every tokenizer call.
- Malformed `<pause...>` fragments raise instead of reaching the tokenizer.
- Leading, trailing, zero, negative, and nonfinite pauses are rejected.
- Adjacent pauses at the same character boundary are merged.
- Explicit and inline plans cannot control the same batch item.
- Batch plans must match batch size; `None` leaves an item uncontrolled.
- Pause duration converts to 25 Hz codec frames with round-half-up semantics.

### Frame Planning

- Speech duration is estimated once from full cleaned text.
- `speed` changes speech frames only.
- Without `duration`, final frames equal speech frames plus pause frames.
- With `duration`, pauses consume part of the requested final total.
- Phrase boundaries follow cumulative duration-estimator weights.
- Allocation reserves at least one speech frame for every nonempty phrase.
- Controlled long-form items are rejected; uncontrolled long-form items remain
  valid in a mixed batch.

### Diffusion and Fixed Tokens

The pause layout becomes a target token template. Pause codec tokens are fixed;
speech positions remain masked. Conditional and unconditional classifier-free
guidance branches receive the same template.

The iterative sampler was changed to:

- derive schedules from each item's real mask count;
- skip items with no remaining masks;
- clamp `topk` to remaining masks;
- exclude fixed positions from sampling;
- update conditional and unconditional target slices together;
- check fixed tokens after every diffusion step;
- preserve the original all-mask schedule, sampling order, and random-number
  behavior for uncontrolled items.

This last property was verified against pristine `0.2.0`: seeded uncontrolled
output remained exactly equal at the codec-token level (`8 x 129`, zero token
differences).

### Postprocessing

Uncontrolled output follows the original postprocessing path. Controlled output
keeps internal silence instead of collapsing it, while retaining edge trimming,
normalization, fades, and edge padding. No waveform silence is inserted after
generation.

## Single-Reference Narration Controls

### Public API

Install the optional local aligner:

```bash
pip install -e ".[narration]"
```

Create one prompt and reuse it:

```python
import soundfile as sf
import torch

from omnivoice import NarrationController, OmniVoice

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)
prompt = model.create_voice_clone_prompt(
    "ref.wav",
    "Exact transcript of the permanent reference recording.",
)
controller = NarrationController(model)

result = controller.generate(
    'The result looked ordinary, but it was '
    '<emphasis strength="strong">not</emphasis> ordinary.',
    voice_clone_prompt=prompt,
    num_step=128,
    candidates=3,
)
sf.write("controlled.wav", result.audio, result.sample_rate)
```

Supported controls:

```text
<rate value="0.85">slower phrase</rate>
<rate value="1.20">faster phrase</rate>
<emphasis strength="reduced">de-emphasized words</emphasis>
<emphasis strength="moderate">important words</emphasis>
<emphasis strength="strong">critical words</emphasis>
<intonation type="rising">rising ending</intonation>
<intonation type="falling">falling ending</intonation>
<aside>parenthetical delivery</aside>
<pause:0.80>
```

Optional shorthand is enabled only with `shorthand=True`:

- `**text**`: moderate emphasis.
- `***text***`: strong emphasis.
- `(text)`: aside delivery.
- `-- text --` written with em dashes in input: isolated strong emphasis.

Explicit markup is safer when parentheses or dashes are literal prose.

### How Local Control Works

`NarrationController` generates a marker-free baseline, aligns expected words,
and maps each character span to codec frames. It then rebuilds only the selected
span plus five transition frames on each side.

Every codec token outside the local inpaint window stays fixed. The controller
generates multiple candidates and ranks them using:

- word-sequence preservation;
- requested duration for rate controls;
- prominence and duration proxies for emphasis and aside;
- pitch direction for intonation;
- fixed-token integrity.

Final audio is decoded from the final codec tensor. The system does not splice
waveforms and does not switch voice references.

### Parser Rules

- Controls cannot nest or overlap.
- Empty controls are rejected.
- Rate values must remain between `0.7` and `1.4`.
- Pause markers cannot sit inside a regional control.
- Malformed tags and unsupported attributes raise before model generation.
- The current controller supports one English short-form item per call.

Rate is the most deterministic control. Emphasis, aside, and intonation depend
on candidate generation and acoustic ranking. The requested effect can remain
subtle even when automated checks pass.

## CMU Pronunciation Conditioning

Legacy full-codebook regeneration sometimes changed pronunciation while giving
the strongest emphasis. CMU ARPABET conditioning was added to retain that
freedom while guiding the controlled word's phonemes.

```python
result = controller.generate(
    'This was <emphasis strength="strong">unexpected</emphasis>.',
    voice_clone_prompt=prompt,
    pronunciations={
        "unexpected": "AH2 N IH0 K S P EH1 K T IH0 D",
    },
    num_step=128,
    candidates=3,
)
```

Behavior:

- Values accept ARPABET with or without brackets.
- Vowels require stress `0`, `1`, or `2`; consonants reject stress.
- Normal text remains the source of offsets, alignment, and expected ASR words.
- Bracketed phonemes are used only for internal generation conditioning.
- Pronunciations apply both inside and outside regional controls.
- Pause offsets are remapped after phoneme substitution.
- A pause cannot split a pronunciation-controlled word.

Informal listening during development found CMU conditioning fixed the cited
`unexpected`, `genuine`, and `nobody` failures while preserving useful emphasis.
The tracked automated reports still label perceptual pronunciation and identity
approval as manual because ASR and waveform metrics cannot prove them.

## Rejected Protected-Codebook Experiment

A separate `experiment/pronunciation-safe-controls` branch tested preserving one
to four early codec codebooks inside the controlled window. It kept original
word duration and reduced phonetic rewriting, but it also removed most useful
emphasis. Listening found the new outputs very close to baseline.

Decision: do not include protected-codebook safe mode. Keep legacy full-codebook
regeneration and use explicit CMU conditioning when pronunciation needs help.

## Generation Traces and Native Diagnostics

Trace mode records per codebook and codec frame:

- unmask step;
- selected token log probability;
- entropy;
- classifier-free-guidance delta;
- fixed-token mask;
- pause-token mask;
- final tokens.

These traces support debugging and future editors without changing generated
audio. `narration_capabilities()` exposes a stable, model-free schema for the
available controls and trace fields.

`scripts/native_generation_capture.py` captures a standalone diagnostic bundle
without Whisper or external alignment. It records:

- duration-estimator word locations;
- punctuation boundaries;
- decoded RMS and pitch per 40 ms codec frame;
- token uncertainty and decode order;
- codec similarity to tokenizer-encoded digital silence;
- inferred pause candidates;
- raw and postprocessed audio plus hashes and runtime metadata.

These values describe model generation evidence, not semantic intent. Native
tokens cannot reliably say that a phrase was an aside or that a word was
emphasized. Natural pause candidates are inferred from energy and silence-token
similarity; they are not exact model annotations.

## Validation

### Automated tests

```bash
.venv/bin/python -m pytest -q
```

Coverage includes parsing, frame math, batch behavior, fixed-token templates,
diffusion schedules, postprocessing, uncontrolled compatibility, narration
candidate selection, CMU validation, pause remapping, and trace integrity.

### Pause smoke tests

```bash
.venv/bin/python scripts/pause_smoke.py --preset quick --num-step 32
.venv/bin/python scripts/pause_smoke.py \
  --preset acceptance --num-step 128 --two-pass-fallback \
  --report reports/pause_smoke_results.json
```

### Narration smoke tests

```bash
.venv/bin/python scripts/narration_control_smoke.py \
  --preset quick --num-step 32 --candidates 1
.venv/bin/python scripts/narration_control_smoke.py \
  --preset acceptance --num-step 128 --candidates 3 \
  --report reports/narration_control_results.json
```

### CMU matrix

```bash
.venv/bin/python scripts/cmu_pronunciation_smoke.py \
  --num-step 128 --candidates 3
```

Generated WAVs remain ignored under `outputs/`. Tracked JSON reports contain
settings, revisions, hashes, transcripts, timing, memory, and acoustic metrics.

## Results and Decisions

### Native pause control

One-pass full acceptance passed only the primary 0.80-second case. The 0.40,
1.20, and secondary-reference 0.80 cases missed the strict 120 ms incremental
pause tolerance after postprocessing. Two-pass boundary inpainting repaired two
of three failures; the primary 1.20 case still missed by 160 ms.

Decision: fixed-token pauses are technically real, but the complete strict
acceptance matrix failed. Keep pause control experimental. Do not claim exact
perceptual duration from codec-frame count alone.

### Narration controls

The 128-step, three-candidate matrix preserved normalized word sequence and
fixed codec tokens in all cases. Slow and fast rate hit duration targets. Aside,
moderate emphasis, strong emphasis, rising intonation, and falling intonation
all produced measurable local changes.

Decision: single-reference narration control is technically viable. Rate and
aside are strongest objectively. Emphasis and intonation remain listening-gated.

### CMU conditioning

Automated cases preserved expected words for `unexpected`, `nobody`, and native
bracket variants of `genuine`. CMU changed the balance among duration, pitch,
and loudness instead of guaranteeing louder emphasis.

Decision: CMU conditioning is the preferred pronunciation guard for legacy
full-codebook emphasis. Use verified CMUdict entries and keep human review.

## Known Limits

- Short-form English only for `NarrationController`.
- Local faster-whisper alignment is required for narration controls.
- No automatic ASR or fallback inside `OmniVoice.generate()`.
- No automatic semantic detection of emphasis or aside.
- No waveform concatenation.
- No reference switching.
- Exact pause frame count does not guarantee exact perceived pause duration.
- ASR word preservation does not prove correct pronunciation, identity, clicks,
  transition naturalness, or intended emotional delivery.
- Human listening remains the final acceptance gate.

## File Map

- `omnivoice/controls.py`: pause parsing, validation, and frame layout.
- `omnivoice/models/omnivoice.py`: public pause API, templates, diffusion, trace
  capture, and controlled postprocessing.
- `omnivoice/narration.py`: markup, shorthand, local inpainting, candidate
  ranking, CMU conditioning, reports, and capability schema.
- `scripts/pause_smoke.py`: pause acceptance and two-pass fallback.
- `scripts/narration_control_smoke.py`: narration-control A/B matrix.
- `scripts/cmu_pronunciation_smoke.py`: pronunciation comparison matrix.
- `scripts/native_generation_capture.py`: native trace and pause evidence bundle.
- `reports/*.json`: tracked reproducibility and acceptance data.
- `findings.md`: concise experiment decisions.

