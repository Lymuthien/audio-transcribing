import unittest
from unittest.mock import MagicMock

from audio_transcribing.builders import VoiceSeparatorDirector
from audio_transcribing.interfaces import ResamplingVoiceSeparator


class TestVoiceSeparatorDirector(unittest.TestCase):
    def setUp(self):
        self.mock_separator = MagicMock(spec=ResamplingVoiceSeparator)
        self.mock_separator.get_audio_stream.return_value = (
            b"audio_data", 44100
        )
        self.mock_separator.get_mono_audio.return_value = b"mono_audio"
        self.mock_separator.resample_audio.return_value = b"resampled_audio"

        self.speakers = (
            {"start": 0, "end": 5, "speaker": 1},
            {"start": 6, "end": 10, "speaker": 2},
        )
        self.mock_separator.separate_speakers.return_value = self.speakers

        self.director = VoiceSeparatorDirector(self.mock_separator)

    def test_set_separator(self):
        self.director.set_separator(self.mock_separator)

    def test_set_separator_with_invalid_type(self):
        with self.assertRaises(TypeError):
            self.director.set_separator("invalid_separator")

    def test_separate_speakers(self):
        content = b"sample_audio_content"
        result = self.director.separate_speakers(
            content, max_speakers=2
        )

        self.mock_separator.get_audio_stream.assert_called_once_with(
            content
        )
        self.mock_separator.get_mono_audio.assert_called_once_with(
            b"audio_data"
        )
        self.mock_separator.resample_audio.assert_called_once_with(
            b"mono_audio", 44100
        )
        self.mock_separator.separate_speakers.assert_called_once_with(
            b"resampled_audio", 2
        )

        self.assertEqual(result, self.speakers)

    def test_separate_speakers_with_no_max_speakers(self):
        content = b"sample_audio_content"
        result = self.director.separate_speakers(content)

        self.mock_separator.separate_speakers.assert_called_once_with(
            b"resampled_audio", None
        )

        self.assertEqual(result, self.speakers)


if __name__ == "__main__":
    unittest.main()
