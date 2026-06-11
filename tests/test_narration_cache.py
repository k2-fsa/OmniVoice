import tempfile
import unittest
import importlib.util

import numpy as np

from omnivoice.narration.cache import NarrationCache
from omnivoice.narration.generator import assemble_from_plan
from omnivoice.narration.schema import NarrationPlan, NarrationSegment


class NarrationCacheTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("soundfile") is None, "soundfile not installed")
    def test_cache_key_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = NarrationCache(tmp)
            segment = NarrationSegment(
                id="s1",
                index=0,
                text="Teste de narração.",
                pause_after_ms=700,
                speed=0.92,
            )
            key = cache.make_key(
                segment=segment,
                model_name="model",
                voice_mode="design",
                generation_settings={"num_step": 4},
                voice_identity="voice",
            )
            path = cache.put_cached_segment(
                key,
                np.zeros(240, dtype=np.float32),
                24000,
                {"text": segment.text},
            )
            cached = cache.get_cached_segment(key)

            self.assertTrue(path.exists())
            self.assertIsNotNone(cached)
            audio, sample_rate, audio_path = cached
            self.assertEqual(sample_rate, 24000)
            self.assertEqual(audio_path, path)
            self.assertGreaterEqual(len(audio), 240)

    def test_assemble_blocks_audio_paths_outside_cache(self):
        with tempfile.TemporaryDirectory() as cache_root:
            with tempfile.NamedTemporaryFile(suffix=".wav") as outside:
                segment = NarrationSegment(
                    id="s1",
                    index=0,
                    text="Teste seguro.",
                    pause_after_ms=0,
                    speed=1.0,
                    audio_path=outside.name,
                )
                plan = NarrationPlan(preset="Presentation", segments=[segment])

                with self.assertRaises(ValueError):
                    assemble_from_plan(
                        plan,
                        sample_rate=24000,
                        cache=NarrationCache(cache_root),
                    )


if __name__ == "__main__":
    unittest.main()
