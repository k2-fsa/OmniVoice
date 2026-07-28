import pytest

from omnivoice.text_normalization import (
    CandidateSpan,
    normalize_candidates,
    normalize_from_ner,
)
from omnivoice.utils.text import normalize_text


def span(text, surface, label, *, occurrence=0, **kwargs):
    positions = [index for index in range(len(text)) if text.startswith(surface, index)]
    start = positions[occurrence]
    return CandidateSpan(start, start + len(surface), label, surface, **kwargs)


@pytest.mark.parametrize(
    ("text", "surface", "label", "spoken"),
    [
        ("Có 25 sách.", "25", "CARDINAL", "hai mươi lăm"),
        ("Nhiệt độ 27,5.", "27,5", "DECIMAL", "hai mươi bảy phẩy năm"),
        ("Năm 2026.", "2026", "YEAR", "hai nghìn không trăm hai mươi sáu"),
        ("Hẹn 27/07/2026.", "27/07/2026", "DATE", "ngày hai mươi bảy tháng bảy"),
        ("Lúc 08:30.", "08:30", "TIME", "tám giờ ba mươi phút"),
        ("Giá 250.000đ.", "250.000đ", "MONEY", "hai trăm năm mươi nghìn đồng"),
        ("Nặng 2.5 kg.", "2.5 kg", "MEASUREMENT", "hai phẩy năm ki lô gam"),
        ("Tăng 12,5%.", "12,5%", "PERCENT", "mười hai phẩy năm phần trăm"),
        ("Gọi 0912 345 678.", "0912 345 678", "PHONE", "không chín một hai"),
        ("Xong 1/2.", "1/2", "FRACTION", "một phần hai"),
        ("Tỉ số 3-1.", "3-1", "SCORE", "ba một"),
        ("Đứng thứ 2.", "2", "ORDINAL", "thứ hai"),
    ],
)
def test_every_label(text, surface, label, spoken):
    result = normalize_candidates(text, [span(text, surface, label)])
    assert spoken in result.text


def test_observed_leading_whitespace_span_preserved():
    text = "Tôi có 25 quyển sách."
    result = normalize_candidates(text, [CandidateSpan(6, 9, "CARDINAL", " 25")])
    assert result.text == "Tôi có hai mươi lăm quyển sách."


def test_identifier_context_overrides_phone_and_keeps_zeroes():
    text = "Mã sinh viên của tôi là 3036123456."
    candidate = span(text, "3036123456", "PHONE")
    result = normalize_candidates(text, [candidate])
    assert result.text.endswith("ba không ba sáu một hai ba bốn năm sáu.")
    assert result.diagnostics[0].effective_label == "IDENTIFIER"


@pytest.mark.parametrize("model_label", ["CARDINAL", "ORDINAL", "PHONE"])
def test_room_context_overrides_numeric_model_labels_to_identifier(model_label):
    text = "Phòng 105."
    result = normalize_candidates(text, [span(text, "105", model_label)])
    assert result.text == "Phòng một không năm."


def test_invalid_mismatch_unknown_and_parse_failure_preserve_exact_text():
    text = "Giữ 31/02/2026 và 1.234."
    candidates = [
        span(text, "31/02/2026", "DATE"),
        span(text, "1.234", "DECIMAL"),
        CandidateSpan(-1, 4, "CARDINAL"),
        CandidateSpan(4, 6, "CARDINAL", "99"),
        span(text, "1.234", "ALIEN"),
    ]
    assert normalize_candidates(text, candidates).text == text


def test_bare_single_digit_grouping_is_ambiguous_even_if_labeled_cardinal():
    text = "Giá trị là 1.234."
    assert normalize_candidates(
        text,
        [span(text, "1.234", "CARDINAL")],
    ).text == text


def test_right_to_left_repeated_entities_and_punctuation():
    text = "25, rồi 25!"
    result = normalize_candidates(
        text, [span(text, "25", "CARDINAL", occurrence=0), span(text, "25", "CARDINAL", occurrence=1)]
    )
    assert result.text == "hai mươi lăm, rồi hai mươi lăm!"


def test_overlap_prefers_score_then_length():
    text = "Giá 250.000đ."
    low = span(text, "250.000", "CARDINAL", score=0.1)
    high = span(text, "250.000đ", "MONEY", score=0.9)
    result = normalize_candidates(text, [low, high])
    assert result.text == "Giá hai trăm năm mươi nghìn đồng."
    assert any(item.reason.startswith("overlap") for item in result.diagnostics)


def test_duplicate_spans_are_deterministic_and_replace_only_once():
    text = "Có 25 sách."
    duplicate = span(text, "25", "CARDINAL", score=0.8)
    result = normalize_candidates(text, [duplicate, duplicate])
    assert result.text == "Có hai mươi lăm sách."
    assert sum(item.action == "normalized" for item in result.diagnostics) == 1


def test_identical_inputs_produce_identical_result_and_diagnostics():
    text = "Có 25 sách, giá 250.000đ."
    candidates = [
        span(text, "25", "CARDINAL", score=0.8),
        span(text, "250.000đ", "MONEY", score=0.9),
    ]
    assert normalize_candidates(text, candidates) == normalize_candidates(
        text,
        candidates,
    )


@pytest.mark.parametrize("score", ["bad", float("nan"), float("inf")])
def test_invalid_scores_preserve_source(score):
    text = "Có 25 sách."
    candidate = span(text, "25", "CARDINAL", score=score)
    assert normalize_candidates(text, [candidate]).text == text


def test_unicode_spaces_and_dash_variants_keep_unrelated_characters():
    text = "Tỉ\u00a0số 3–1; còn −25."
    result = normalize_candidates(
        text,
        [
            span(text, "3–1", "SCORE"),
            span(text, "−25", "CARDINAL"),
        ],
    )
    assert result.text == "Tỉ\u00a0số ba một; còn âm hai mươi lăm."


def test_long_detector_failure_preserves_complete_input():
    text = ("Đoạn Unicode αβγ có 25 mục. " * 500).strip()

    def unsupported_length(_text):
        raise OverflowError("token limit")

    assert normalize_from_ner(text, unsupported_length).text == text


def test_normalized_output_is_idempotent_without_new_candidates():
    text = "Có 25 sách."
    first = normalize_candidates(text, [span(text, "25", "CARDINAL")]).text
    assert normalize_candidates(first, []).text == first


def test_detector_failure_is_sentence_level_safe():
    def broken(_):
        raise RuntimeError("model unavailable")

    result = normalize_from_ner("Giữ 007.", broken)
    assert result.text == "Giữ 007."


def test_public_normalize_text_accepts_an_injected_detector():
    text = "\u00a0Tôi có 2 hộp.\u202f"
    calls = []

    def detector(value):
        calls.append(value)
        start = value.index("2")
        return [
            {
                "start": start,
                "end": start + 1,
                "label": "CARDINAL",
                "text": "2",
            }
        ]

    assert normalize_text(text, "vi", detector=detector) == (
        "\u00a0Tôi có hai hộp.\u202f"
    )
    assert calls == [text]


def test_public_normalize_text_preserves_complete_input_if_normalizer_raises(
    monkeypatch,
):
    text = "\u00a0Có 25 hộp và 30 túi.\u202f"
    detector_calls = []

    def detector(value):
        detector_calls.append(value)
        return [
            {
                "start": value.index(surface),
                "end": value.index(surface) + len(surface),
                "label": "CARDINAL",
                "text": surface,
            }
            for surface in ("25", "30")
        ]

    def failing_normalizer(value, active_detector):
        assert value == text
        assert len(active_detector(value)) == 2
        raise RuntimeError("unexpected normalizer failure")

    monkeypatch.setattr(
        "omnivoice.text_normalization.normalize_from_ner",
        failing_normalizer,
    )
    assert normalize_text(text, "vi", detector=detector) == text
    assert detector_calls == [text]


def test_unexpected_pipeline_failure_preserves_complete_input(monkeypatch):
    text = "Có 25 hộp và 30 túi."
    candidates = [
        span(text, "25", "CARDINAL"),
        span(text, "30", "CARDINAL"),
    ]

    def fail_on_second(value, label, *, surface=""):
        if surface == "30":
            raise RuntimeError("unexpected verbalizer failure")
        return "hai mươi lăm"

    monkeypatch.setattr(
        "omnivoice.text_normalization.pipeline.verbalize",
        fail_on_second,
    )
    result = normalize_candidates(text, candidates)
    assert result.text == text
    assert result.diagnostics[0].reason.startswith("pipeline failure:")


@pytest.mark.parametrize("bad", ["", "abc", "1,2,3", "\x00", "—", "9" * 100])
def test_malformed_candidates_never_crash(bad):
    text = f"A{bad}Z"
    result = normalize_candidates(text, [CandidateSpan(1, 1 + len(bad), "DECIMAL")])
    assert isinstance(result.text, str)
