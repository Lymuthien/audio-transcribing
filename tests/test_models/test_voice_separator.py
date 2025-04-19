import unittest
from unittest.mock import patch, Mock

from audio_transcribing.models.voice_separator import VoiceSeparatorWithPyAnnote


class TestVoiceSeparator(unittest.TestCase):
    @patch("pyannote.audio.Pipeline.from_pretrained", return_value=Mock())
    def setUp(self, pipelane: Mock):
        self.voice_separator = VoiceSeparatorWithPyAnnote('')

    def test_transcription_with_invalid_audio(self):
        invalid_audio = "Not a numpy array"
        with self.assertRaises(AttributeError):
            self.voice_separator.separate_speakers(invalid_audio)


if __name__ == '__main__':
    unittest.main()
