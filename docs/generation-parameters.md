# Generation Parameters

Parameters can be passed as keyword arguments to `model.generate(...)` or via the `OmniVoiceGenerationConfig` dataclass. See below for the full list and which category each belongs to.

```python
# 1) Direct keyword arguments
audio = model.generate(text="Hello world", num_step=32, guidance_scale=2.0)

# 2) Via OmniVoiceGenerationConfig dataclass
from omnivoice import OmniVoiceGenerationConfig

config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
audio = model.generate(text="Hello world", generation_config=config)
```

## Decoding

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_step` | int | 32 | Number of iterative unmasking steps. Higher values improve quality but slow down generation. Use 16 for faster inference. |
| `denoise` | bool | True | Prepend the `<|denoise|>` token to the input, which signals the model to produce cleaner speech. |
| `guidance_scale` | float | 2.0 | Classifier-free guidance scale.|
| `t_shift` | float | 0.1 | Time-step shift for the noise schedule. Smaller values emphasise earlier steps in decoding. |

## Sampling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `position_temperature` | float | 5.0 | Temperature for mask-position selection. 0 = greedy (deterministic). Higher values increase randomness. |
| `class_temperature` | float | 0.0 | Temperature for token sampling at each step. 0 = greedy (deterministic). Higher values increase randomness. |
| `layer_penalty_factor` | float | 5.0 | Penalty applied to deeper codebook layers, encouraging earlier (lower) layers to unmask first. |

## Duration & Speed

These accept a single value applied to all items, or a per-item list (useful in batch mode):

```python
# Request a 10-second pre-synthesis audio-token budget
audio = model.generate(text="Hello, this is a test of duration control", duration=10.0)

# Faster speech (1.2x faster than estimated)
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `duration` | float or list[float \| None] | None | Positive, finite pre-synthesis audio-token budget in seconds. Overrides `speed` when set. Post-processing can change the physical WAV duration. |
| `speed` | float or list[float \| None] | None | Speed factor. Values > 1.0 produce shorter audio (faster); values < 1.0 produce longer audio (slower). Ignored when `duration` is set. Defaults to 1.0 when both are None. |

Priority: `duration` > `speed`.

> **Note:** `duration` controls the number of audio tokens generated; it is not a hard guarantee for the final waveform length. Silence removal can shorten the decoded waveform, while `pad_duration` adds silence to both edges. For an untrimmed, unpadded waveform, use `postprocess_output=False`, `pad_duration=0`, and optionally `fade_duration=0`, then measure the physical result.

The duration budget is converted at the audio tokenizer's frame rate. The
conversion retains floor semantics for fractional token counts, but a binary64
product exactly one ULP below an integer is treated as that integer. This avoids
losing a token for values such as `duration=29 / 25` at a 25 Hz tokenizer while
leaving genuinely fractional budgets unchanged. Audio-token counts are
tokenizer-specific implementation details, so `duration` remains the public
control instead of exposing a separate token-count parameter.

## Pre/Post Processing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preprocess_prompt` | bool | True | Whether to apply preprocessing to the voice-clone prompt audio (remove long silences in reference audio, add punctuation in the end of reference text). |
| `postprocess_output` | bool | True | Shorten long internal silences and trim leading/trailing silence. Padding and fades are controlled independently below. |
| `output_min_silence_ms` | int | 500 | Minimum internal silence duration to shorten, in milliseconds. Set to 0 to skip internal-silence shortening; edge trimming remains controlled by `postprocess_output`. |
| `output_keep_silence_ms` | int or None | None | Maximum total silence retained from each shortened internal gap, in milliseconds. `None` uses `output_min_silence_ms`. |
| `output_lead_silence_ms` | int | 100 | Leading silence retained before optional padding, in milliseconds. |
| `output_trail_silence_ms` | int | 100 | Trailing silence retained before optional padding, in milliseconds. |
| `output_peak_limit` | float or None | None | Optional final absolute peak ceiling in `(0, 1]`. The complete waveform is scaled only when its peak exceeds this value. |
| `output_target_lead_silence_ms` | int or None | None | Optional exact final leading silence in milliseconds. Zero-valued PCM16 proxy samples at the leading edge are replaced with this amount of digital silence after fades and generic padding. |
| `output_target_trail_silence_ms` | int or None | None | Optional exact final trailing silence in milliseconds. Zero-valued PCM16 proxy samples at the trailing edge are replaced with this amount of digital silence after fades and generic padding. |
| `pad_duration` | float | 0.1 | Silence padding duration per side in seconds. Set to 0 to disable. |
| `fade_duration` | float | 0.1 | Fade-in/out curve duration in seconds. Set to 0 to disable. |

`output_min_silence_ms` is the detection threshold, while
`output_keep_silence_ms` is the maximum total gap retained after detection.
Keeping these controls separate prevents a silence from being retained once on
each side of a split and becoming twice as long as requested.

Silence detection uses a PCM proxy, but the selected ranges are sliced from
the original floating-point waveform. This avoids quantizing or clipping
voiced samples merely because silence post-processing is enabled. Set
`output_peak_limit` when the final output must stay below a PCM encoder's
full-scale ceiling; limiting preserves duration and relative dynamics.

The `output_target_*_silence_ms` controls are exact alignment targets supplied
by the caller. They do not infer timing from reference audio. `None` preserves
the corresponding processed edge byte for byte, while `0` removes all digital
silence on that side. Edge detection is sample-accurate on a PCM16 proxy and
preserves PCM-representable low-level attacks instead of using a dBFS silence
threshold. Numeric targets take precedence over `pad_duration` on
their respective side. When fitting a fixed external window, subtract the
intentional leading and trailing targets from the `duration` token budget; the
targets extend the physical waveform after synthesis and do not time-stretch
spoken samples.

All millisecond controls require non-negative integers; padding and fade
durations require finite non-negative numbers. The `output_*` controls affect
generated audio only. Prompt preprocessing keeps its historical silence
retention so existing voice-clone prompts do not change identity as a side
effect of output post-processing configuration.

## Long-Form Generation

To support stable long-form speech generation with low VRAM consumption, the text is automatically split into smaller segments when the estimated duration of the generated speech exceeds `audio_chunk_duration`, with each segment producing approximately `audio_chunk_duration` seconds of audio. This approach allows the model to accept arbitrarily long text and generate arbitrarily long speech with near-constant VRAM consumption.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | Target chunk duration (seconds) when splitting long text. |
| `audio_chunk_threshold` | float | 30.0 | Estimated audio duration (seconds) above which chunking is activated. |
