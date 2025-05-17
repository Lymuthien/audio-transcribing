import unittest
from unittest.mock import patch, Mock

from audio_transcribing.infrastructure.faster_whisper_processor import FasterWhisperProcessor


class TestFasterWhisperProcessor(unittest.TestCase):

    @patch("faster_whisper.WhisperModel")
    def setUp(self, whisper_model: Mock):
        self.processor = FasterWhisperProcessor(model_size='base')

    def test_transcription_with_invalid_audio(self):
        invalid_audio = "Not a numpy array"
        with self.assertRaises(AttributeError):
            self.processor.transcribe_audio(invalid_audio)


if __name__ == '__main__':
    unittest.main()
