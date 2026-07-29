import os
from pathlib import Path

import pytest

from omnivoice.text_normalization import (
    DEFAULT_MODEL_PATH,
    get_bamibert_detector,
    normalize_candidates,
)
from omnivoice.text_normalization.detector import _reset_detector_cache


@pytest.mark.integration
def test_real_bamibert_candidate_pipeline_semantics():
    """Exercise real weights only inside this explicitly marked test."""
    if os.environ.get("OMNIVOICE_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("set OMNIVOICE_RUN_REAL_MODEL_TESTS=1 to load real weights")
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.is_dir():
        pytest.skip(f"BamiBERT model directory is absent: {model_path}")

    _reset_detector_cache()
    detector = get_bamibert_detector(model_path, "cpu")
    cases = (
        (
            "Tôi có 25 quyển sách.",
            {"CARDINAL"},
            ("hai mươi lăm",),
        ),
        (
            "Hẹn lúc 08:30 ngày 27/07/2026.",
            {"TIME", "DATE"},
            ("tám giờ ba mươi phút", "hai mươi bảy tháng bảy", "năm hai nghìn"),
        ),
        (
            "Phí tham gia là 250.000 đồng.",
            {"MONEY"},
            ("hai trăm năm mươi nghìn đồng",),
        ),
        (
            "Gọi cho tôi qua số 0912 345 678.",
            {"PHONE"},
            ("không chín một hai ba bốn năm sáu bảy tám",),
        ),
        (
            "Mã sinh viên của tôi là 3036123456.",
            {"PHONE", "IDENTIFIER"},
            ("ba không ba sáu một hai ba bốn năm sáu",),
        ),
    )

    for text, expected_labels, expected_fragments in cases:
        candidates = detector(text)
        assert candidates, text
        assert all(
            text[item.start : item.end] == item.text
            and 0 <= item.start < item.end <= len(text)
            for item in candidates
        )
        assert expected_labels.intersection({item.label for item in candidates}), (
            text,
            candidates,
        )
        result = normalize_candidates(text, candidates)
        for fragment in expected_fragments:
            assert fragment in result.text, (text, candidates, result)
