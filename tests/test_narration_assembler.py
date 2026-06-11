import unittest

import numpy as np

from omnivoice.narration.assembler import assemble_segments, audio_duration_seconds


class NarrationAssemblerTest(unittest.TestCase):
    def test_assemble_adds_real_silence_duration(self):
        sample_rate = 1000
        first = np.ones(100, dtype=np.float32) * 0.2
        second = np.ones(200, dtype=np.float32) * 0.2

        audio = assemble_segments([first, second], [500, 1000], sample_rate=sample_rate)

        self.assertAlmostEqual(audio_duration_seconds(audio, sample_rate), 1.8, places=2)
        self.assertEqual(len(audio), 1800)


if __name__ == "__main__":
    unittest.main()
