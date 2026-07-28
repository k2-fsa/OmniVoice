import sys
import threading
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from omnivoice.text_normalization.detector import (
    BamiBertDetector,
    DetectorConfig,
    DetectorLoadError,
    _reset_detector_cache,
    configure_bamibert,
    convert_predictions,
    get_bamibert_detector,
)


@pytest.fixture(autouse=True)
def reset_detector():
    _reset_detector_cache()
    yield
    _reset_detector_cache()


def test_import_does_not_construct_transformers_pipeline():
    with patch("transformers.pipeline") as pipeline:
        importlib.import_module("omnivoice.text_normalization.detector")
    pipeline.assert_not_called()


def test_importing_package_and_cli_modules_does_not_load_weights():
    with patch("transformers.pipeline") as pipeline:
        import omnivoice
        from omnivoice.cli import demo, infer, infer_batch

        importlib.reload(omnivoice)
        importlib.reload(infer)
        importlib.reload(infer_batch)
        importlib.reload(demo)
    pipeline.assert_not_called()


def test_importing_real_model_scripts_does_not_load_weights():
    with patch("transformers.pipeline") as pipeline:
        from scripts import (
            benchmark_text_normalization,
            try_text_normalizer,
        )

        importlib.reload(benchmark_text_normalization)
        importlib.reload(try_text_normalizer)
    pipeline.assert_not_called()


def test_removed_legacy_package_has_no_live_import_path():
    assert importlib.util.find_spec("omnivoice.utils.vietnamese_normalization") is None


def test_prediction_conversion_binds_original_offsets_and_id_alias():
    text = "Mã 007."
    candidates = convert_predictions(
        text,
        [{"start": 3, "end": 6, "entity_group": "ID", "score": 0.9}],
    )
    assert candidates[0].text == "007"
    assert candidates[0].label == "IDENTIFIER"


def test_prediction_conversion_maps_unit_alias_to_measurement():
    candidates = convert_predictions(
        "2.5 kg",
        [{"start": 0, "end": 6, "entity_group": "UNIT", "score": 0.9}],
    )
    assert candidates[0].label == "MEASUREMENT"


def test_money_prediction_expands_to_adjacent_currency_marker():
    text = "Phí là 250.000 đồng."
    candidates = convert_predictions(
        text,
        [{"start": 6, "end": 14, "entity_group": "MONEY", "score": 0.8}],
    )
    assert candidates[0].text == " 250.000 đồng"
    assert text[candidates[0].start : candidates[0].end] == candidates[0].text


@pytest.mark.parametrize(
    "output",
    [
        None,
        "not predictions",
        [{}],
        [{"start": -1, "end": 2, "entity_group": "CARDINAL"}],
        [{"start": 0, "end": 99, "entity_group": "CARDINAL"}],
        [{"start": 0, "end": 1}],
    ],
)
def test_malformed_prediction_output_fails_conversion(output):
    with pytest.raises((TypeError, ValueError)):
        convert_predictions("1", output)


def test_same_configuration_constructs_once_and_reuses():
    created = []

    def factory(config):
        detector = object()
        created.append((config, detector))
        return detector

    first = get_bamibert_detector("model-a", "cpu", factory=factory)
    assert get_bamibert_detector("model-a", "cpu", factory=factory) is first
    assert get_bamibert_detector("model-a", "cpu", factory=factory) is first
    assert len(created) == 1


def test_material_configuration_change_replaces_bounded_cache():
    created = []

    def factory(config):
        created.append(config)
        return object()

    first = get_bamibert_detector("model-a", "cpu", factory=factory)
    second = get_bamibert_detector("model-a", "cuda:0", factory=factory)
    third = get_bamibert_detector("model-b", "cuda:0", factory=factory)
    assert len({id(first), id(second), id(third)}) == 3
    assert len(created) == 3


def test_concurrent_initialization_constructs_exactly_once():
    calls = 0
    calls_lock = threading.Lock()

    def factory(_config):
        nonlocal calls
        with calls_lock:
            calls += 1
        return object()

    with ThreadPoolExecutor(max_workers=12) as pool:
        detectors = list(
            pool.map(
                lambda _: get_bamibert_detector("model-a", "cpu", factory=factory),
                range(24),
            )
        )
    assert calls == 1
    assert all(item is detectors[0] for item in detectors)


def test_failed_configuration_is_not_retried_per_request():
    calls = 0

    def broken(_config):
        nonlocal calls
        calls += 1
        raise OSError("missing weights")

    with pytest.raises(DetectorLoadError, match="missing weights"):
        get_bamibert_detector("broken", "cpu", factory=broken)
    with pytest.raises(DetectorLoadError, match="missing weights"):
        get_bamibert_detector("broken", "cpu", factory=broken)
    assert calls == 1


def test_failed_configuration_does_not_poison_a_valid_configuration():
    def broken(_config):
        raise OSError("missing weights")

    with pytest.raises(DetectorLoadError):
        get_bamibert_detector("broken", "cpu", factory=broken)

    valid = object()
    assert (
        get_bamibert_detector(
            "valid",
            "cpu",
            factory=lambda _config: valid,
        )
        is valid
    )


def test_wrapper_uses_eval_inference_mode_and_serializes_predictions():
    model = Mock()
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def pipeline(text):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        with state_lock:
            active -= 1
        return [{"start": 0, "end": len(text), "entity_group": "CARDINAL"}]

    pipeline.model = model
    contexts = 0

    class InferenceMode:
        def __enter__(self):
            nonlocal contexts
            contexts += 1

        def __exit__(self, *_args):
            return False

    detector = BamiBertDetector(pipeline, InferenceMode)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(detector, ["1"] * 16))
    model.eval.assert_called_once_with()
    assert contexts == 16
    assert max_active == 1


def test_config_resolves_repository_relative_default(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_BAMIBERT_MODEL", raising=False)
    config = DetectorConfig.resolve()
    assert config.model_path.endswith("artifacts/models/bamibert_augmented_best")
    assert config.device == "cpu"


def test_process_configuration_uses_the_detector_source_of_truth(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_BAMIBERT_MODEL", raising=False)
    monkeypatch.delenv("OMNIVOICE_BAMIBERT_DEVICE", raising=False)
    configure_bamibert("configured-model", "cpu-test")
    config = DetectorConfig.resolve()
    assert config.model_path.endswith("configured-model")
    assert config.device == "cpu-test"
