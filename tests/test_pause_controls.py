import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omnivoice import OmniVoice, PausePlan, PauseSpec
from omnivoice.controls import (
    PauseLayout,
    canonicalize_pause_plan,
    create_pause_layout,
    parse_pause_markers,
    pause_seconds_to_frames,
)
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig


def test_inline_markers_strip_and_preserve_structured_offsets():
    cleaned, plan = parse_pause_markers(
        "First  <pause:0.40>   second<pause:0.80>third."
    )
    assert cleaned == "First secondthird."
    assert plan == PausePlan((PauseSpec(5, 0.4), PauseSpec(12, 0.8)))
    assert cleaned[: plan.pauses[0].after_char] == "First"


def test_adjacent_markers_merge_at_same_offset():
    cleaned, plan = parse_pause_markers("Hello <pause:0.40> <pause:0.80> world")
    assert cleaned == "Hello world"
    assert len(plan.pauses) == 1
    assert plan.pauses[0].after_char == 5
    assert plan.pauses[0].seconds == pytest.approx(1.2)


@pytest.mark.parametrize(
    "text",
    [
        "<pause:0.4>Hello",
        "Hello<pause:0.4>",
        "Hello<pause:0>world",
        "Hello<pause:-1>world",
        "Hello<pause:nan>world",
        "Hello<pause:inf>world",
        "Hello<pause: 0.4>world",
        "Hello<pause:abc>world",
        "Hello<pause:0.4world",
        "Hello<pause foo>world",
    ],
)
def test_invalid_inline_markers_raise(text):
    with pytest.raises(ValueError):
        parse_pause_markers(text)


def test_explicit_plan_sorts_merges_and_rejects_edges():
    plan = canonicalize_pause_plan(
        "alpha beta gamma",
        PausePlan((PauseSpec(10, 0.2), PauseSpec(5, 0.4), PauseSpec(5, 0.1))),
    )
    assert plan.pauses == (PauseSpec(5, 0.5), PauseSpec(10, 0.2))
    for offset in (0, len("alpha beta gamma")):
        with pytest.raises(ValueError):
            canonicalize_pause_plan(
                "alpha beta gamma", PausePlan((PauseSpec(offset, 0.4),))
            )


def test_half_up_frame_conversion():
    assert pause_seconds_to_frames(0.02) == 1
    assert pause_seconds_to_frames(0.06) == 2
    assert pause_seconds_to_frames(0.40) == 10
    with pytest.raises(ValueError):
        pause_seconds_to_frames(0.001)


def test_speed_changes_only_speech_frames():
    plan = PausePlan((PauseSpec(5, 0.4),))
    normal = create_pause_layout("alpha beta", plan, 100, 1.0, None, 25, len)
    fast = create_pause_layout("alpha beta", plan, 100, 2.0, None, 25, len)
    assert normal.speech_frames == 100
    assert fast.speech_frames == 50
    assert normal.pause_frames == fast.pause_frames == (10,)
    assert normal.total_frames == 110
    assert fast.total_frames == 60


def test_duration_is_final_total_and_pause_consumes_it():
    plan = PausePlan((PauseSpec(5, 0.8),))
    layout = create_pause_layout("alpha beta", plan, 100, 3.0, 2.0, 25, len)
    assert layout.total_frames == 50
    assert layout.pause_frames == (20,)
    assert layout.speech_frames == 30


def test_duration_rejects_too_few_phrase_frames():
    plan = PausePlan((PauseSpec(1, 0.4), PauseSpec(2, 0.4)))
    with pytest.raises(ValueError, match="one speech frame"):
        create_pause_layout("abc", plan, 10, 1.0, 0.84, 25, len)


def test_cumulative_weight_allocation_is_monotonic_and_exact():
    plan = PausePlan((PauseSpec(1, 0.4), PauseSpec(3, 0.4)))
    weights = {"a": 1.0, "bb": 2.0, "cccc": 4.0}
    layout = create_pause_layout(
        "abbcccc", plan, 70, 1.0, None, 25, lambda text: weights[text]
    )
    assert layout.phrase_frames == (10, 20, 40)
    assert sum(layout.phrase_frames) == layout.speech_frames


class _FakeAudioTokenizer:
    device = torch.device("cpu")
    config = SimpleNamespace(frame_rate=25)

    def __init__(self):
        self.waveforms = []

    def encode(self, waveform):
        self.waveforms.append(waveform.clone())
        frames = waveform.shape[-1] // 960
        codes = torch.arange(8).view(1, 8, 1).expand(1, 8, frames).clone()
        return SimpleNamespace(audio_codes=codes)


class _TemplateHarness:
    device = torch.device("cpu")
    sampling_rate = 24000
    config = SimpleNamespace(num_audio_codebook=8, audio_mask_id=1024)

    def __init__(self):
        self.audio_tokenizer = _FakeAudioTokenizer()

    _encode_pause_tokens = OmniVoice._encode_pause_tokens
    _create_pause_template = OmniVoice._create_pause_template
    _load_pause_source = OmniVoice._load_pause_source


class _PreprocessHarness(_TemplateHarness):
    duration_estimator = SimpleNamespace(calculate_total_weight=lambda text: len(text))

    _preprocess_all = OmniVoice._preprocess_all
    _normalize_pause_plans = OmniVoice._normalize_pause_plans
    _ensure_list = OmniVoice._ensure_list

    def _estimate_target_tokens(self, text, ref_text, ref_tokens, speed=1.0):
        return max(1, int(20 / speed))


def test_preprocess_batch_requires_matching_plan_list_and_supports_none_items():
    model = _PreprocessHarness()
    plan = PausePlan((PauseSpec(5, 0.4),))
    with pytest.raises(ValueError, match="requires a pause_plan list"):
        model._preprocess_all(["alpha beta", "plain text"], pause_plan=plan)
    with pytest.raises(ValueError, match="match text batch size"):
        model._preprocess_all(["alpha beta", "plain text"], pause_plan=[plan])

    task = model._preprocess_all(["alpha beta", "plain text"], pause_plan=[plan, None])
    assert task.controlled == [True, False]
    assert task.target_lens == [30, 20]
    assert task.pause_spans == [((10, 20),), ()]
    assert task.target_templates[1] is None


def test_preprocess_rejects_explicit_plan_with_inline_marker():
    model = _PreprocessHarness()
    with pytest.raises(ValueError, match="Cannot combine"):
        model._preprocess_all(
            "alpha<pause:0.4> beta",
            pause_plan=PausePlan((PauseSpec(5, 0.4),)),
        )


def test_digital_silence_template_spans_all_eight_codebooks():
    model = _TemplateHarness()
    layout = PauseLayout(5, 15, (2, 3), (10,))
    template, spans = model._create_pause_template(layout, None, {})
    assert template.shape == (8, 15)
    assert spans == ((2, 12),)
    assert torch.all(template[:, :2] == 1024)
    assert torch.all(template[:, 12:] == 1024)
    assert torch.equal(template[:, 2:12], torch.arange(8).view(8, 1).expand(8, 10))
    assert model.audio_tokenizer.waveforms[0].shape == (1, 1, 10 * 960)
    assert torch.count_nonzero(model.audio_tokenizer.waveforms[0]) == 0


def test_room_tone_is_mono_center_cropped_and_validated():
    model = _TemplateHarness()
    stereo = torch.stack((torch.arange(20000), torch.arange(20000) + 2)).float()
    source = model._load_pause_source((stereo, 24000), 10)
    tokens = model._encode_pause_tokens(10, source)
    encoded = model.audio_tokenizer.waveforms[-1]
    assert tokens.shape == (8, 10)
    assert encoded.shape[-1] == 9600
    expected = stereo.mean(0)[(20000 - 9600) // 2 : (20000 + 9600) // 2]
    assert torch.equal(encoded[0, 0], expected)
    with pytest.raises(ValueError, match="shorter"):
        model._load_pause_source((torch.zeros(100), 24000), 10)
    bad = torch.zeros(10000)
    bad[0] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        model._load_pause_source((bad, 24000), 10)


class _DiffusionHarness:
    device = torch.device("cpu")
    config = SimpleNamespace(num_audio_codebook=2, audio_mask_id=4)

    def __init__(self):
        self.inputs = []

    def _prepare_inference_inputs(
        self,
        text,
        num_target_tokens,
        ref_text,
        ref_audio_tokens,
        lang,
        instruct,
        denoise,
        target_template,
    ):
        target = (
            target_template.clone()
            if target_template is not None
            else torch.full((2, num_target_tokens), 4, dtype=torch.long)
        )
        prefix = torch.zeros((2, 2), dtype=torch.long)
        ids = torch.cat((prefix, target), dim=1).unsqueeze(0)
        mask = torch.zeros((1, ids.shape[-1]), dtype=torch.bool)
        mask[:, 2:] = True
        return {"input_ids": ids, "audio_mask": mask}

    def __call__(self, input_ids, audio_mask, attention_mask):
        self.inputs.append(input_ids.clone())
        logits = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
        return SimpleNamespace(logits=logits)

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        shape = c_logits.shape[:-1]
        pred = torch.ones(shape, dtype=torch.long)
        scores = torch.arange(pred.numel(), dtype=torch.float32).view(shape)
        return pred, scores

    _generate_iterative = OmniVoice._generate_iterative


def _diffusion_task(template):
    return GenerationTask(
        batch_size=1,
        texts=["test"],
        target_lens=[template.shape[-1]],
        langs=[None],
        instructs=[None],
        ref_texts=[None],
        ref_audio_tokens=[None],
        ref_rms=[None],
        target_templates=[template],
        pause_spans=[((1, 3),)],
        controlled=[True],
    )


def test_cfg_inputs_share_template_and_fixed_tokens_never_change():
    model = _DiffusionHarness()
    template = torch.full((2, 5), 4, dtype=torch.long)
    template[:, 1:3] = torch.tensor([[2, 3], [1, 2]])
    output = model._generate_iterative(
        _diffusion_task(template),
        OmniVoiceGenerationConfig(
            num_step=3, position_temperature=0.0, layer_penalty_factor=0.0
        ),
    )[0]
    first_input = model.inputs[0]
    assert torch.equal(first_input[0, :, -5:], first_input[1, :, :5])
    fixed = template != 4
    assert torch.equal(output[fixed], template[fixed])
    assert torch.all(output[~fixed] != 4)


def test_mixed_mask_counts_keep_schedule_and_topk_safe():
    model = _DiffusionHarness()
    controlled = torch.tensor([[4, 2, 4], [4, 2, 4]])
    uncontrolled = torch.full((2, 5), 4, dtype=torch.long)
    task = GenerationTask(
        batch_size=2,
        texts=["a", "b"],
        target_lens=[3, 5],
        langs=[None, None],
        instructs=[None, None],
        ref_texts=[None, None],
        ref_audio_tokens=[None, None],
        ref_rms=[None, None],
        target_templates=[controlled, uncontrolled],
        pause_spans=[((1, 2),), ()],
        controlled=[True, False],
    )
    outputs = model._generate_iterative(
        task,
        OmniVoiceGenerationConfig(
            num_step=8, position_temperature=0.0, layer_penalty_factor=0.0
        ),
    )
    assert [tuple(output.shape) for output in outputs] == [(2, 3), (2, 5)]
    assert all(torch.all(output != 4) for output in outputs)


def test_controlled_postprocessing_preserves_internal_silence():
    model = SimpleNamespace(sampling_rate=24000)
    tone = np.full((1, 12000), 0.2, dtype=np.float32)
    silence = np.zeros((1, 48000), dtype=np.float32)
    audio = np.concatenate((tone, silence, tone), axis=-1)
    config = OmniVoiceGenerationConfig(pad_duration=0.0, fade_duration=0.0)
    uncontrolled = OmniVoice._post_process_audio(model, audio, 0.1, config, False)
    controlled = OmniVoice._post_process_audio(model, audio, 0.1, config, True)
    assert controlled.shape[-1] > uncontrolled.shape[-1] + 12000


def test_public_api_appends_pause_parameters_after_existing_parameters():
    names = list(inspect.signature(OmniVoice.generate).parameters)
    assert names[:11] == [
        "self",
        "text",
        "language",
        "ref_text",
        "ref_audio",
        "voice_clone_prompt",
        "instruct",
        "duration",
        "speed",
        "generation_config",
        "pause_plan",
    ]
    assert names[11:] == ["pause_audio", "kwargs"]
