"""Regressions for fractional-millisecond endpoints of silence detection."""

import numpy as np
import pytest

from omnivoice.utils.audio import remove_silence


@pytest.mark.parametrize("sample_count", [24001, 24011, 24013, 24103])
@pytest.mark.parametrize("mid_sil", [0, 500])
def test_active_tail_is_not_rounded_to_milliseconds(sample_count, mid_sil):
    audio = np.full((1, sample_count), 0.25, dtype=np.float32)
    output = remove_silence(audio, 24000, mid_sil=mid_sil)
    np.testing.assert_array_equal(output, audio)


def test_retained_middle_gap_does_not_discard_fractional_tail():
    audio = np.concatenate(
        [
            np.full((1, 4800), 0.25, dtype=np.float32),
            np.zeros((1, 24000), dtype=np.float32),
            np.full((1, 4811), 0.5, dtype=np.float32),
        ],
        axis=-1,
    )
    output = remove_silence(audio, 24000, mid_sil=500, keep_mid_sil=160)
    assert np.count_nonzero(output == np.float32(0.5)) == 4811


@pytest.mark.parametrize("mid_sil", [0, 500])
def test_preservation_keeps_quiet_attack_and_release(mid_sil):
    quiet = np.full((1, 4800), 0.0002, dtype=np.float32)
    audio = np.concatenate(
        [quiet, np.full((1, 12000), 0.25, dtype=np.float32), quiet], axis=-1
    )
    legacy = remove_silence(audio, 24000, mid_sil=mid_sil, lead_sil=50, trail_sil=50)
    assert legacy.shape[-1] < audio.shape[-1]
    output = remove_silence(
        audio,
        24000,
        mid_sil=mid_sil,
        lead_sil=50,
        trail_sil=50,
        preserve_active_edges=True,
    )
    np.testing.assert_array_equal(output, audio)


def test_preservation_keeps_an_entirely_quiet_nonzero_utterance():
    audio = np.full((1, 24011), 0.0002, dtype=np.float32)
    output = remove_silence(audio, 24000, preserve_active_edges=True)
    np.testing.assert_array_equal(output, audio)


def test_preservation_still_shortens_internal_silence():
    quiet = np.full((1, 4800), 0.0002, dtype=np.float32)
    loud = np.full((1, 4800), 0.25, dtype=np.float32)
    audio = np.concatenate(
        [quiet, loud, np.zeros((1, 24000), dtype=np.float32), loud, quiet], axis=-1
    )
    output = remove_silence(audio, 24000, keep_mid_sil=160, preserve_active_edges=True)
    assert output.shape[-1] < audio.shape[-1]
    np.testing.assert_array_equal(output[output != 0], audio[audio != 0])


def test_preservation_flag_is_strict_and_opt_in():
    from omnivoice import OmniVoiceGenerationConfig

    assert OmniVoiceGenerationConfig().output_preserve_active_edges is False
    assert (
        OmniVoiceGenerationConfig(
            output_preserve_active_edges=True
        ).output_preserve_active_edges
        is True
    )
    with pytest.raises(TypeError):
        OmniVoiceGenerationConfig(output_preserve_active_edges="false")
    with pytest.raises(TypeError):
        remove_silence(
            np.ones((1, 24), dtype=np.float32), 24000, preserve_active_edges=1
        )
