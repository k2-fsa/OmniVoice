from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


class RecordingModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, audio_path):
        path = Path(audio_path)
        self.calls.append(path)
        return self.scores[path.name]


def test_select_korean_wavs_filters_multilingual_test_list(tmp_path):
    from omnivoice.eval.mos.komos import select_korean_wavs

    samples = [
        {"id": "english_00000", "language_id": "en"},
        {"id": "korean_00000", "language_id": "ko"},
        {"id": "korean_00001", "language_id": "ko-KR"},
    ]

    assert select_korean_wavs(samples, tmp_path, "flac") == [
        tmp_path / "korean_00000.flac",
        tmp_path / "korean_00001.flac",
    ]


def test_select_korean_wavs_rejects_lists_without_korean_rows(tmp_path):
    from omnivoice.eval.mos.komos import select_korean_wavs

    with pytest.raises(ValueError, match="no Korean samples"):
        select_korean_wavs(
            [{"id": "english_00000", "language_id": "en"}], tmp_path, "wav"
        )


def test_score_korean_wavs_skips_missing_files_and_averages_valid_scores(tmp_path):
    from omnivoice.eval.mos.komos import score_korean_wavs

    first = tmp_path / "korean_00000.wav"
    second = tmp_path / "korean_00001.wav"
    missing = tmp_path / "korean_00002.wav"
    first.touch()
    second.touch()
    model = RecordingModel({first.name: 3.5, second.name: 4.5})

    results, errors = score_korean_wavs([first, missing, second], model)

    assert results == [(first, 3.5), (second, 4.5)]
    assert errors == [(missing, f"File not found: {missing}")]
    assert model.calls == [first, second]


def test_score_korean_wavs_propagates_model_failures(tmp_path):
    from omnivoice.eval.mos.komos import score_korean_wavs

    wav_file = tmp_path / "korean_00000.wav"
    wav_file.touch()

    class BrokenModel:
        def score(self, audio_path):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        score_korean_wavs([wav_file], BrokenModel())


def test_write_results_emits_per_file_and_average(tmp_path):
    from omnivoice.eval.mos.komos import write_results

    output = tmp_path / "nested" / "komos.log"
    write_results(
        output,
        [(Path("korean_00000.wav"), 3.5), (Path("korean_00001.wav"), 4.5)],
    )

    assert output.read_text(encoding="utf-8") == (
        "Name\tKOMOS\n"
        "korean_00000.wav\t3.500\n"
        "korean_00001.wav\t4.500\n"
        "\nAverage KOMOS: 4.000 (2 samples)\n"
    )


def test_load_model_uses_pinned_revision_and_selected_device(monkeypatch):
    from omnivoice.eval.mos import komos

    captured = {}

    class FakeKOMOS:
        @classmethod
        def from_pretrained(cls, model_id, *, revision):
            captured.update(model_id=model_id, revision=revision)
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    monkeypatch.setitem(sys.modules, "komos", SimpleNamespace(KOMOS=FakeKOMOS))

    model = komos.load_model("cpu")

    assert isinstance(model, FakeKOMOS)
    assert captured == {
        "model_id": "upskyy/komos",
        "revision": "c558983b9777d4f01aa61003b151dea37db0cc2d",
        "device": "cpu",
        "eval": True,
    }
