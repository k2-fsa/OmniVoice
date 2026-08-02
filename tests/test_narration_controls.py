import pytest
import torch
from types import SimpleNamespace
import numpy as np

from omnivoice.narration import (
    NarrationControl,
    _aligned_control_audio,
    _conditioned_text,
    _core_frame_target,
    _normalize_cmu_pronunciation,
    _resolve_pronunciations,
    _shift_fixed_spans,
    _transcribe,
    align_expected_words,
    build_inpaint_template,
    control_word_indices,
    parse_narration_controls,
    NarrationController,
    narration_capabilities,
)


def test_explicit_controls_clean_text_and_keep_offsets():
    plan = parse_narration_controls(
        'This was <emphasis strength="strong">not</emphasis> ordinary. '
        '<rate value="0.85">Read this slowly.</rate> '
        '<intonation type="falling">That was final?</intonation>'
    )
    assert plan.text == "This was not ordinary. Read this slowly. That was final?"
    assert [
        (item.kind, plan.text[item.start_char : item.end_char])
        for item in plan.controls
    ] == [
        ("emphasis", "not"),
        ("rate", "Read this slowly."),
        ("intonation", "That was final?"),
    ]
    assert [item.value for item in plan.controls] == ["strong", 0.85, "falling"]


def test_pause_and_narration_controls_share_cleaned_text():
    plan = parse_narration_controls(
        "First.<pause:0.60> <aside>Apparently this mattered.</aside>"
    )
    assert plan.text == "First. Apparently this mattered."
    assert plan.pause_plan.pauses[0].after_char == 6
    assert plan.text[plan.controls[0].start_char : plan.controls[0].end_char] == (
        "Apparently this mattered."
    )


def test_repeated_surface_maps_control_by_original_position():
    plan = parse_narration_controls("foo <emphasis>foo</emphasis>")
    control = plan.controls[0]
    assert (control.start_char, control.end_char) == (4, 7)
    assert plan.text[control.start_char : control.end_char] == "foo"


def test_shorthand_is_optional_and_maps_to_controls():
    source = "This was **not** normal. (At least for now.) The answer — absolutely — changed."
    plain = parse_narration_controls(source, shorthand=False)
    assert plain.controls == ()
    assert plain.text == source

    plan = parse_narration_controls(source, shorthand=True)
    assert [(item.kind, item.value, item.source) for item in plan.controls] == [
        ("emphasis", "moderate", "shorthand"),
        ("aside", "parenthetical", "shorthand"),
        ("emphasis", "strong", "dash"),
    ]
    assert (
        plan.text
        == "This was not normal. At least for now. The answer absolutely changed."
    )


@pytest.mark.parametrize(
    "text",
    [
        '<rate value="0.2">Too fast.</rate>',
        '<rate value="abc">Bad.</rate>',
        '<emphasis strength="extreme">No.</emphasis>',
        '<intonation type="flat">No.</intonation>',
        '<aside unknown="x">No.</aside>',
        "<emphasis>missing close",
        "<emphasis><aside>nested</aside></emphasis>",
        "<aside>text<pause:0.4></aside> after",
    ],
)
def test_invalid_controls_raise(text):
    with pytest.raises(ValueError):
        parse_narration_controls(text)


def test_control_word_indices_uses_character_overlap():
    text = "The hidden timing changed everything."
    start = text.index("hidden")
    control = NarrationControl(
        "emphasis", start, start + len("hidden timing"), "strong"
    )
    assert control_word_indices(text, control) == (1, 2)


def test_expected_word_alignment_maps_matching_blocks():
    expected = ["the", "hidden", "timing", "changed"]
    asr = [
        {"normalized": "the"},
        {"normalized": "hidden"},
        {"normalized": "timing"},
        {"normalized": "changed"},
    ]
    assert align_expected_words(expected, asr) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_inpaint_template_preserves_every_outside_token():
    tokens = torch.arange(2 * 20).view(2, 20)
    template, old_window, new_core = build_inpaint_template(
        tokens, 8, 12, 6, mask_id=99, transition_frames=2
    )
    assert old_window == (6, 14)
    assert new_core == (8, 14)
    assert template.shape == (2, 22)
    assert torch.equal(template[:, :6], tokens[:, :6])
    assert torch.all(template[:, 6:16] == 99)
    assert torch.equal(template[:, 16:], tokens[:, 14:])


def test_control_frame_targets_match_requested_behavior():
    assert _core_frame_target(NarrationControl("rate", 0, 1, 1.25), 20) == 16
    assert _core_frame_target(NarrationControl("rate", 0, 1, 0.8), 20) == 25
    assert _core_frame_target(NarrationControl("emphasis", 0, 1, "moderate"), 20) == 23
    assert _core_frame_target(NarrationControl("emphasis", 0, 1, "strong"), 20) == 27
    assert _core_frame_target(NarrationControl("intonation", 0, 1, "rising"), 20) == 20


def test_conditioned_text_adds_model_cues_without_changing_surface_words():
    text = "This was not ordinary."
    start = text.index("not")
    strong = NarrationControl("emphasis", start, start + 3, "strong")
    assert _conditioned_text(text, strong) == "This was —NOT— ordinary."
    rising = NarrationControl("intonation", 0, len(text), "rising")
    assert _conditioned_text(text, rising) == "This was not ordinary?"


def test_conditioned_text_can_use_cmu_without_changing_alignment_text():
    text = "This was unexpected."
    start = text.index("unexpected")
    control = NarrationControl("emphasis", start, start + 10, "strong")
    pronunciation = "[AH2 N IH0 K S P EH1 K T IH0 D]"
    assert _conditioned_text(text, control, pronunciation) == (
        "This was —[AH2 N IH0 K S P EH1 K T IH0 D]—."
    )


def test_cmu_pronunciation_normalization_and_control_resolution():
    assert (
        _normalize_cmu_pronunciation("ah2 n ih0 k s p eh1 k t ih0 d")
        == "[AH2 N IH0 K S P EH1 K T IH0 D]"
    )
    plan = parse_narration_controls(
        'This was <emphasis strength="strong">unexpected</emphasis>.'
    )
    assert _resolve_pronunciations(
        plan,
        {"unexpected": "AH2 N IH0 K S P EH1 K T IH0 D"},
    ) == {0: "[AH2 N IH0 K S P EH1 K T IH0 D]"}


def test_cmu_pronunciation_can_protect_word_inside_multiword_control():
    plan = parse_narration_controls(
        'This was <emphasis strength="strong">an unexpected result</emphasis>.'
    )
    resolved = _resolve_pronunciations(
        plan,
        {"unexpected": "AH2 N IH0 K S P EH1 K T IH0 D"},
    )
    assert resolved == {
        0: ((3, 13, "[AH2 N IH0 K S P EH1 K T IH0 D]"),)
    }
    assert _conditioned_text(plan.text, plan.controls[0], resolved[0]) == (
        "This was —an [AH2 N IH0 K S P EH1 K T IH0 D] result—."
    )


def test_generate_plan_rejects_non_plan_before_loading_model():
    controller = NarrationController(SimpleNamespace())
    with pytest.raises(TypeError, match="NarrationPlan"):
        controller.generate_plan("not a plan")


def test_narration_capabilities_are_stable_and_defensively_copied():
    first = narration_capabilities()
    first["regional_controls"]["emphasis"]["values"].append("fake")
    second = narration_capabilities()
    assert second["frame_rate"] == 25
    assert second["regional_controls"]["emphasis"]["values"] == [
        "reduced",
        "moderate",
        "strong",
    ]


@pytest.mark.parametrize(
    "value",
    ["", "[AH2 N", "AH N", "N1 OW1", "XX1"],
)
def test_invalid_cmu_pronunciations_raise(value):
    with pytest.raises((TypeError, ValueError)):
        _normalize_cmu_pronunciation(value)


def test_pronunciation_key_must_match_controlled_surface():
    plan = parse_narration_controls(
        'This was <emphasis strength="strong">unexpected</emphasis>.'
    )
    with pytest.raises(ValueError, match="controlled span"):
        _resolve_pronunciations(plan, {"nobody": "N OW1 B AA2 D IY2"})


def test_fixed_pause_spans_shift_or_reject_overlap():
    assert _shift_fixed_spans(((3, 5), (15, 18)), (8, 12), 7) == ((3, 5), (18, 21))
    with pytest.raises(ValueError, match="fixed pause"):
        _shift_fixed_spans(((9, 11),), (8, 12), 7)


def test_numpy_audio_is_resampled_to_whisper_sample_rate():
    class FakeAligner:
        def transcribe(self, audio, **kwargs):
            assert len(audio) == 16000
            return [], SimpleNamespace(language="en", language_probability=1.0)

    result = _transcribe(FakeAligner(), np.zeros(24000, dtype=np.float32), 24000)
    assert result["words"] == []


def test_intonation_metrics_use_three_word_ending_context():
    text = "The hidden timing changed everything."
    control = NarrationControl("intonation", 0, len(text), "rising")
    words = []
    cursor = 0.0
    for word in ["the", "hidden", "timing", "changed", "everything"]:
        words.append(
            {
                "word": word,
                "normalized": word,
                "start": cursor,
                "end": cursor + 0.2,
                "probability": 1.0,
            }
        )
        cursor += 0.2
    transcript = {"words": words}
    metrics = _aligned_control_audio(
        text,
        control,
        transcript,
        np.zeros(24000, dtype=np.float32),
        24000,
    )
    assert metrics is not None
    assert metrics["first_word_index"] == 2
    assert metrics["last_word_index"] == 4
