"""Per-item ``speed`` handling in generation preprocessing (no model weights needed)."""

from functools import partial
from types import SimpleNamespace

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.duration import RuleDurationEstimator


def _stub_model():
    stub = SimpleNamespace(duration_estimator=RuleDurationEstimator())
    stub._ensure_list = partial(OmniVoice._ensure_list, stub)
    stub._estimate_target_tokens = partial(OmniVoice._estimate_target_tokens, stub)
    return stub


def test_speed_list_with_none_entry_uses_default_speed():
    # generate() documents speed as list[Optional[float]]; None means default.
    stub = _stub_model()
    texts = ["Hello there, how are you?", "Hello there, how are you?"]

    task = OmniVoice._preprocess_all(stub, text=texts, speed=[None, 2.0])

    default = OmniVoice._preprocess_all(stub, text=texts[0])
    assert task.target_lens[0] == default.target_lens[0]
    assert task.speed == [1.0, 2.0]
