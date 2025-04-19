# tests/test_audio_processing_mixin.py

import unittest

import numpy as np
from audio_transcribing.utils.audio_processing_mixin import AudioProcessingMixin


class TestAudioProcessingMixin(unittest.TestCase):
    def test_get_audio_stream_valid_audio(self):
        content = b'\x00\x01\x02\x03'
        with self.assertRaises(Exception):
            AudioProcessingMixin.get_audio_stream(content)

    def test_get_mono_audio_already_mono(self):
        audio = np.array([0.1, 0.2, 0.3])
        result = AudioProcessingMixin.get_mono_audio(audio)
        self.assertTrue(np.array_equal(result, audio))

    def test_get_mono_audio_multi_channel(self):
        audio = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]])
        result = AudioProcessingMixin.get_mono_audio(audio)
        expected = np.array([0.15, 0.3, 0.45])
        self.assertTrue(np.allclose(result, expected))

    def test_resample_audio_16k_sample_rate(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        sample_rate = 16000
        result = AudioProcessingMixin.resample_audio(audio, sample_rate)
        self.assertTrue(np.array_equal(result, audio))

    def test_resample_audio_non_16k_sample_rate(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        sample_rate = 8000
        result = AudioProcessingMixin.resample_audio(audio, sample_rate)
        self.assertEqual(len(result), len(audio) * 2)


if __name__ == "__main__":
    unittest.main()
