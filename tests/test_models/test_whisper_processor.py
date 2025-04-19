import unittest
from unittest.mock import patch, Mock

import numpy as np
from audio_transcribing.models.whisper_processor import WhisperProcessor


class TestWhisperProcessor(unittest.TestCase):

    @patch("whisper.load_model")
    def setUp(self, mock_model: Mock):
        self.whisper_processor = WhisperProcessor(model_size='base')

    @patch("whisper.transcribe")
    def test_transcribe_audio_with_default_language(self, mock_transcribe: Mock):
        mock_transcribe.return_value = {
            'text': 'Test transcription',
            'language': 'en'
        }

        fake_audio = np.random.rand(16000)
        transcription, detected_language = self.whisper_processor.transcribe_audio(fake_audio)

        self.assertEqual(transcription, 'Test transcription')
        self.assertEqual(detected_language, 'en')

    @patch("whisper.transcribe")
    def test_transcribe_audio_with_setting_language(self, mock_transcribe: Mock):
        mock_transcribe.return_value = {
            'text': 'Test transcription',
            'language': 'en'
        }

        fake_audio = np.random.rand(16000)
        transcription, detected_language = self.whisper_processor.transcribe_audio(fake_audio, 'ru')

        self.assertEqual(transcription, 'Test transcription')
        self.assertEqual(detected_language, 'ru')

    def test_transcription_with_invalid_audio(self):
        invalid_audio = "Not a numpy array"
        with self.assertRaises(AttributeError):
            self.whisper_processor.transcribe_audio(invalid_audio)


if __name__ == '__main__':
    unittest.main()
