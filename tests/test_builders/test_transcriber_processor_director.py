import unittest
from unittest.mock import MagicMock

from audio_transcribing.builders import TranscribeProcessorDirector
from audio_transcribing.interfaces import WhisperTranscribeProcessor


class TestTranscribeProcessorDirector(unittest.TestCase):

    def setUp(self):
        self.mock_processor = MagicMock(spec=WhisperTranscribeProcessor)
        self.mock_processor.get_audio_stream.return_value = (
            b"mock_audio", 44100
        )
        self.mock_processor.get_mono_audio.return_value = b"mock_mono_audio"
        self.mock_processor.resample_audio.return_value = b"mock_resampled_audio"
        self.mock_processor.transcribe_audio.return_value = (
            "mock_transcription", "en"
        )

        self.director = TranscribeProcessorDirector(self.mock_processor)

    def test_set_processor(self):
        new_processor = MagicMock(spec=WhisperTranscribeProcessor)
        self.director.set_processor(new_processor)

    def test_set_processor_with_invalid_type(self):
        with self.assertRaises(TypeError):
            self.director.set_processor("invalid_processor")

    def test_transcribe_audio_calls_methods_in_order(self):
        content = b"test_audio_data"
        language = "en"
        main_theme = "test_theme"

        transcription, detected_language = self.director.transcribe_audio(
            content, language, main_theme
        )

        self.mock_processor.get_audio_stream.assert_called_once_with(
            content
        )
        self.mock_processor.get_mono_audio.assert_called_once_with(
            b"mock_audio"
        )
        self.mock_processor.resample_audio.assert_called_once_with(
            b"mock_mono_audio", 44100
        )
        self.mock_processor.transcribe_audio.assert_called_once_with(
            b"mock_resampled_audio", language, main_theme
        )

        self.assertEqual(transcription, "mock_transcription")
        self.assertEqual(detected_language, "en")

    def test_transcribe_audio_without_optional_params(self):
        content = b"test_audio_data"

        transcription, detected_language = self.director.transcribe_audio(
            content
        )

        self.mock_processor.get_audio_stream.assert_called_once_with(
            content
        )
        self.mock_processor.get_mono_audio.assert_called_once_with(
            b"mock_audio"
        )
        self.mock_processor.resample_audio.assert_called_once_with(
            b"mock_mono_audio", 44100
        )
        self.mock_processor.transcribe_audio.assert_called_once_with(
            b"mock_resampled_audio", None, None
        )

        self.assertEqual(transcription, "mock_transcription")
        self.assertEqual(detected_language, "en")


if __name__ == "__main__":
    unittest.main()
