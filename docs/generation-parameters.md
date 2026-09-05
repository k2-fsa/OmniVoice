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

# Fit the final waveform to exactly 240,000 sample frames per channel
audio = model.generate(
    text="Hello, this is a test of physical duration control",
    final_duration_samples=240_000,
)

# Faster speech (1.2x faster than estimated)
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `duration` | float or list[float \| None] | None | Positive, finite pre-synthesis audio-token budget in seconds. Overrides `speed` when set. Post-processing can change the physical WAV duration. |
| `speed` | float or list[float \| None] | None | Speed factor. Values > 1.0 produce shorter audio (faster); values < 1.0 produce longer audio (slower). Ignored when `duration` is set. Defaults to 1.0 when both are None. |
| `final_duration` | float or list[float \| None] | None | Positive, finite physical output duration in seconds. Converted at the model sample rate using the canonical decimal representation and decimal HALF_UP rounding. Independent of `duration`. Mutually exclusive with `final_duration_samples` per item. |
| `final_duration_samples` | int or list[int \| None] | None | Authoritative positive integer output length in sample frames per channel. No seconds conversion is performed. Independent of `duration`. Mutually exclusive with `final_duration` per item. |

Priority: `duration` > `speed`.

The physical controls are orthogonal to that priority: they never change the
audio-token budget, speaking speed, or chunking decision. A scalar applies to
every item; an exact-length iterable may mix values with `None`. When external
timing already has an integer frame count, use `final_duration_samples` so no
floating-point intent has to be reconstructed.

> **Note:** `duration` controls the number of audio tokens generated; it is not a hard guarantee for the final waveform length. Silence removal can shorten the decoded waveform, while `pad_duration` adds silence to both edges. For the waveform emitted directly by the codec decoder, use `output_mode="raw_codec"`, then measure the physical result.

The duration budget is converted at the audio tokenizer's frame rate. The
conversion retains floor semantics for fractional token counts, but a binary64
product exactly one ULP below an integer is treated as that integer. This avoids
losing a token for values such as `duration=29 / 25` at a 25 Hz tokenizer while
leaving genuinely fractional budgets unchanged. Audio-token counts are
tokenizer-specific implementation details, so `duration` remains the public
control instead of exposing a separate token-count parameter.

Physical framing runs after the selected output pipeline. Underflow is always
filled by appending exact floating-point zeros after the complete source, even
when the trailing edge has an explicit pre-framing anchor. The entire source,
its onset, and both existing edges remain bit-exact; the deficit is reported as
trailing outer-container fill after the authenticated trailing region. Overflow
removes only contiguous frames that are exactly zero in every
channel, preferring the trailing edge; protected anchor regions are never
shortened, and generation fails if reaching the target would cut any active or
protected sample. It never time-stretches, resamples, inserts internal silence,
or truncates speech.

`output_target_*_silence_ms` therefore defines edge authority **before** final
physical framing. It authenticates the generated pre-framing edge samples; it
does not claim that a later outer-container fill will leave the same total
detectable silence at the file boundary. Telemetry reports that fill separately
as `padded_leading_samples` or `padded_trailing_samples`.

`final_duration` is a convenience input. Its canonical decimal text is
converted to an exact integer ratio, scaled by the integer model sample rate,
and rounded with positive `ROUND_HALF_UP` semantics using integer arithmetic.
The result is independent of the process-global `decimal` context. A float
whose intended rational value was already lost by earlier arithmetic cannot be
reconstructed; use `final_duration_samples` for authoritative sample-accurate
timing.

## Pre/Post Processing

For quiet attacks and releases, opt in to `output_preserve_active_edges=True`
(CLI: `--output_preserve_active_edges`). During silence removal this preserves
every nonzero outer-edge sample, including samples below the -50 dBFS detector
threshold, while allowing internal silence shortening. It defaults to `False`
for compatibility. It also preserves quiet noise; it is not a speech detector.
An entirely quiet but nonzero waveform is retained instead of becoming empty.
The independent fade and PCM16-proxy alignment stages still apply afterwards.
This setting neither regenerates a missing phoneme nor guarantees correct text.

Ranges that reach the end of the silence-detection proxy now retain the exact
physical endpoint. A fractional-millisecond active tail is never discarded
merely because the proxy rounded the total duration to milliseconds.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_mode` | `"processed"` or `"raw_codec"` | `"processed"` | Select the backward-compatible output pipeline or return the codec decoder waveform without output transformations. |
| `preprocess_prompt` | bool | True | Whether to apply preprocessing to the voice-clone prompt audio (remove long silences in reference audio, add punctuation in the end of reference text). |
| `postprocess_output` | bool | True | Shorten long internal silences and trim leading/trailing silence. Padding and fades are controlled independently below. |
| `output_min_silence_ms` | int | 500 | Minimum internal silence duration to shorten, in milliseconds. Set to 0 to skip internal-silence shortening; edge trimming remains controlled by `postprocess_output`. |
| `output_keep_silence_ms` | int or None | None | Maximum total silence retained from each shortened internal gap, in milliseconds. An explicit value is a total. `None` preserves historical pydub per-side behavior and keeps up to `2 * output_min_silence_ms` in total. |
| `output_lead_silence_ms` | int | 100 | Leading silence retained before optional padding, in milliseconds. |
| `output_trail_silence_ms` | int | 100 | Trailing silence retained before optional padding, in milliseconds. |
| `output_peak_limit` | float or None | None | Optional final absolute peak ceiling in `(0, 1]`. The complete waveform is scaled only when its peak exceeds this value. |
| `output_target_lead_silence_ms` | int or None | None | Optional exact pre-framing leading-silence authority in milliseconds. Zero-valued PCM16 proxy samples at the leading edge are replaced with this amount of digital silence after fades and generic padding; final container framing may add separately reported outer fill without changing these authenticated samples. |
| `output_target_trail_silence_ms` | int or None | None | Optional exact pre-framing trailing-silence authority in milliseconds. Zero-valued PCM16 proxy samples at the trailing edge are replaced with this amount of digital silence after fades and generic padding; final container framing may add separately reported outer fill without changing these authenticated samples. |
| `pad_duration` | float | 0.1 | Silence padding duration per side in seconds. Set to 0 to disable. |
| `fade_duration` | float | 0.1 | Fade-in/out curve duration in seconds. Set to 0 to disable. |

`output_min_silence_ms` is the detection threshold. An explicit
`output_keep_silence_ms` is the maximum **total** gap retained after detection.
The `None` default is the compatibility exception: historical OmniVoice passed
the minimum directly to pydub's per-side `keep_silence`, so the equivalent
total is twice the minimum.

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

Before synthesis starts, processed mode converts `pad_duration` and explicit
edge targets to conservative sample counts and rejects values whose intermediate
or final edge allocation cannot fit the platform waveform-size limit. This
preflight cannot know the eventual decoded speech length, but it guarantees that
the configuration-derived edge allocation is representable. Raw codec mode
ignores these post-processing-only allocation controls, as it does at runtime.

### Raw Codec Output

`output_mode="raw_codec"` is an explicit, fail-closed bypass for consumers
that need the waveform emitted by the codec decoder. It bypasses all of the
following operations:

- internal and edge silence removal;
- reference-RMS and peak normalization;
- peak limiting;
- fades and padding;
- exact edge-silence alignment;
- long-form chunk cross-fades.

When a physical duration control is provided, exact output framing is applied
after this bypass. This framing is not signal processing: it only adds exact
zeros or removes exact-zero edge frames, and it fails before cutting an active
sample. With neither control, raw codec output remains sample-identical, but
OmniVoice still validates the decoder contract before reporting success: mono
`(1, T)` input internally, a real floating dtype, at least one sample, and
finite values.

For long-form generation, decoded chunks are concatenated in order without a
cross-fade because `generate()` still returns one waveform per input item.
Prompt preprocessing is an input operation and remains independently
controlled by `preprocess_prompt`.

Raw codec samples are floating-point decoder values and may exceed `[-1, 1]`.
Persisting them without clipping or quantization requires a floating-point
container representation, for example
`soundfile.write(path, audio, sample_rate, subtype="FLOAT")`. Both inference
CLIs use IEEE float WAV automatically when `output_mode="raw_codec"`; processed
mode keeps the historical WAV writer behavior.

`postprocess_output=False` only disables silence removal. It does not bypass
normalization, limiting, fades, padding, or edge alignment, so it is not a
substitute for `raw_codec`. The default `processed` mode retains the historical
pipeline and output behavior.

The same mode is available from both inference CLIs:

```bash
omnivoice-infer --model k2-fsa/OmniVoice --text "Hello" --output out.wav --output_mode raw_codec
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results --output_mode raw_codec
```

### Opt-in Generation Telemetry

Pass `telemetry_callback` to receive one immutable
`OmniVoiceGenerationTelemetry` record after each successful `generate()` call:

```python
from omnivoice import (
    OmniVoiceFramingObservation,
    OmniVoiceGenerationTelemetry,
)


def report(metrics: OmniVoiceGenerationTelemetry) -> None:
    print(f"token generation: {metrics.token_generation_seconds:.3f}s")
    print(f"codec decode: {metrics.codec_decode_seconds:.3f}s")
    print(f"post-processing: {metrics.postprocessing_seconds:.3f}s")
    print(f"output framing: {metrics.output_framing_seconds:.3f}s")
    print(f"framing observer: {metrics.framing_observer_seconds:.3f}s")
    print(f"framing evidence: {metrics.final_duration_items}")
    print(f"wall: {metrics.wall_seconds:.3f}s")


def audit_framing(observation: OmniVoiceFramingObservation) -> None:
    assert observation.source_waveform.shape[0] == 1
    assert observation.final_waveform.shape[0] == 1
    assert not observation.source_waveform.flags.writeable
    assert not observation.final_waveform.flags.writeable
    print(observation.framing)


audio = model.generate(
    text="Hello",
    final_duration_samples=48_000,
    framing_observer=audit_framing,
    telemetry_callback=report,
)
```

The record separates input preparation, prompt/reference preparation when a
prompt is built inside the call, token generation, codec decode,
post-processing, and internal generation wall time. Public callback execution
occurs after the record is finalized and is not included. Prompt preparation is reported as a
measured subset of input preparation. `prompt_preparation_seconds` is `None`
when a reusable prompt is supplied or no reference prompt is built. CUDA
allocated/reserved memory boundary snapshots and per-call peak values are
included only when the model device is CUDA and CUDA is available. Before
resetting PyTorch's process/device-global peak counters, OmniVoice acquires an
exclusive in-process lease for that CUDA device. Telemetry-enabled calls on the
same device are serialized; different devices remain independent. Internal
same-thread reentrancy on the same device fails fast instead of deadlocking.
`memory_allocated_peak_bytes` and `memory_reserved_peak_bytes` cover the complete
synchronized, leased interval through internal output validation. The lease is
released before `framing_observer` or `telemetry_callback` is invoked, so public
callbacks never run under it and may start a new generation safely. Concurrent
uninstrumented CUDA work or external code in the same process can still
contribute to the process/device peak; the value is therefore the true allocator
high-water mark for the documented leased interval, not attribution of every
byte to OmniVoice. Calls without `telemetry_callback` never acquire this lease or
query/reset accelerator memory statistics.

The lease coordinates OmniVoice calls inside one Python process. External code
must not reset PyTorch peak counters for the same device during an instrumented
call; such an out-of-band reset is outside the library's lock domain and would
invalidate the measurement.

Physical-duration work has a separate `output_framing_seconds` stage. Each
requested item also produces an immutable `OmniVoiceFinalDurationTelemetry`
record containing the authoritative target, target source, input/output frame
counts, the explicit operation, protected-edge state, and exact
leading/trailing padding or trimming counts. Calls without a physical target
keep the stage at zero and the item tuple empty.

`framing_observer` is an independent, opt-in integrity hook for consumers that
need to hash or inspect the actual waveform around physical framing. It is
invoked once for each item whose `final_duration` or
`final_duration_samples` value is not `None`, after all structural, numeric,
zero-only, retained-core, authority, and returned-output checks pass. Each
`OmniVoiceFramingObservation` contains:

- immutable mono `(1, T)` real-floating snapshots before and after framing,
  with matching dtypes and finite samples;
- exact retained-source and retained-final slice indices;
- the same authoritative target and operation metadata reported by telemetry.

Padding and trimming are mutually exclusive, so both waveform fields share a
single immutable backing snapshot whose size is the larger of the source and
final waveforms. The other field is a zero-copy slice of that snapshot. This
makes `setflags(write=True)`, direct assignment, and mutation through NumPy's
`.base` chain fail without allocating two complete audit copies. Snapshot
content is compared bit-for-bit, including the retained core and the ndarray
returned by `generate()`; trimmed and padded edge regions must contain numeric
digital zero. A decoder that omits or forges evidence, returns different
same-length samples, or attempts to repair an empty waveform fails before the
user callback or telemetry success callback is emitted.

Consumers must not assume C-contiguous arrays: the zero-copy field may carry a
valid slice stride. Hashing and serialization should walk logical C order (or
bounded logical chunks) rather than hashing unrelated bytes outside the view.

The observer is synchronous. Inspect or hash the arrays during the callback
and avoid retaining them, especially for long-form generation, because a
retained observation also retains its backing waveform snapshot. Observer
callbacks from concurrent generation calls may run concurrently and must be
thread-safe. On CUDA they run only after any generation-telemetry lease has been
released. Exceptions propagate before `generate()` returns.

`framing_observer` does not enable telemetry by itself: with no
`telemetry_callback`, it performs no timer reads, device synchronization, or
accelerator memory queries. When telemetry is enabled,
`framing_observer_seconds` reports internal snapshot construction and evidence
validation separately from `output_framing_seconds`; it excludes the later
public observer callback. Passing an observer with no real
physical target (including an all-`None` target list) is a zero-call no-op.

Telemetry is completely disabled by default: its instrumentation performs no
timer reads, accelerator availability or memory queries, or synchronization.
Enabling it synchronizes CUDA, MPS, or XPU at stage boundaries and therefore
adds measurement overhead. Accelerator synchronization is device-wide: stage
timings are not isolated or reliably attributable to one generation when
concurrent work uses the same CUDA, MPS, or XPU device. Callback invocations
from concurrent generation calls may run concurrently and must be thread-safe;
the library invokes them without holding a global callback lock. If a callback
raises an exception, it is propagated after generation completes.

## Long-Form Generation

To support stable long-form speech generation with low VRAM consumption, the text is automatically split into smaller segments when the estimated duration of the generated speech exceeds `audio_chunk_duration`, with each segment producing approximately `audio_chunk_duration` seconds of audio. This approach allows the model to accept arbitrarily long text and generate arbitrarily long speech with near-constant VRAM consumption.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | Target chunk duration (seconds) when splitting long text. |
| `audio_chunk_threshold` | float | 30.0 | Estimated audio duration (seconds) above which chunking is activated. |
