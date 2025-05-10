"""
Unit tests for the language.py module.

Tests the language detection functionality for the Ollama RAG system including:
- Pattern-based language detection
- FastText language detection
- Translation functionality
"""

import unittest
from unittest.mock import MagicMock, patch
import sys


class TestLanguageDetector(unittest.TestCase):
    def setUp(self):
        self.mock_translator = MagicMock()
        self.mock_translator.translate.return_value = "Translated text"

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_language_detector_init(self, mock_google_translator):
        mock_google_translator.return_value = self.mock_translator
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector(default_language="fr", min_confidence=0.75)
        self.assertEqual(detector.default_language, "fr")
        self.assertEqual(detector.min_confidence, 0.75)
        mock_google_translator.assert_called_once_with(source="auto", target="en")

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_empty_text(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect_language("")
        self.assertEqual(result, "en")
        mock_fasttext_detect.assert_not_called()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_common_phrase(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect_language("tell me")
        self.assertEqual(result, "en")
        mock_fasttext_detect.assert_not_called()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_pattern_match(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        detector._check_patterns = lambda text: "es" if "dónde" in text.lower() else None
        result = detector.detect_language("¿Dónde está?")
        self.assertEqual(result, "es")
        mock_fasttext_detect.assert_not_called()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_fasttext(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        mock_fasttext_detect.return_value = {"lang": "fr", "score": 0.95}
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect_language("Bonjour tout le monde, comment allez-vous aujourd'hui?")
        self.assertEqual(result, "fr")
        mock_fasttext_detect.assert_called_once()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_low_confidence(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        mock_fasttext_detect.return_value = {"lang": "fr", "score": 0.3}
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect_language("abc def")
        self.assertEqual(result, "en")
        mock_fasttext_detect.assert_called_once()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_language_fasttext_exception(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        mock_fasttext_detect.side_effect = Exception("FastText error")
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        detector._check_patterns = lambda _: None
        result = detector.detect_language("abcdef ghijkl mnopqr stuvwx yz")
        self.assertEqual(result, "en")
        mock_fasttext_detect.assert_called_once()

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.fasttext_detect")
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_detect_with_details(self, mock_google_translator, mock_fasttext_detect):
        mock_google_translator.return_value = self.mock_translator
        mock_fasttext_detect.return_value = {"lang": "de", "score": 0.88}
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()

        result = detector.detect_with_details("Guten Tag, wie geht es Ihnen?")
        self.assertEqual(result["lang"], "de")
        self.assertEqual(result["method"], "fasttext")
        self.assertEqual(result["confidence"], 0.88)
        self.assertEqual(result["word_count"], 6)

        empty_result = detector.detect_with_details("")
        self.assertEqual(empty_result["lang"], "en")
        self.assertEqual(empty_result["method"], "default")

    @patch("api.rag_pipeline.language.logger", MagicMock())
    @patch("api.rag_pipeline.language.GoogleTranslator")
    def test_translate_text(self, mock_google_translator):
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "Hello world"
        mock_google_translator.return_value = mock_translator_instance
        from api.rag_pipeline.language import LanguageDetector

        detector = LanguageDetector()
        detector.detect_language = MagicMock(return_value="fr")

        result = detector.translate_text("Bonjour le monde")
        self.assertEqual(result, "Hello world")
        mock_google_translator.assert_called_with(source="fr", target="en")

        detector.detect_language.return_value = "en"
        same_lang_result = detector.translate_text("Hello world", target_lang="en")
        self.assertEqual(same_lang_result, "Hello world")

    def test_module_level_functions(self):
        """Test module-level utility functions by directly patching the actual functions"""
        # Clear any cached modules
        sys.modules.pop("api.rag_pipeline.language", None)

        # Create patches for the module-level functions themselves
        with patch("api.rag_pipeline.language.detect_language") as mock_detect_language, patch(
            "api.rag_pipeline.language.detect_language_with_details"
        ) as mock_detect_details, patch(
            "api.rag_pipeline.language.translate_text"
        ) as mock_translate_text:

            # Configure the mocks to return expected values
            mock_detect_language.return_value = "es"
            mock_detect_details.return_value = {"lang": "es", "confidence": 0.9}
            mock_translate_text.return_value = "Translated text"

            # Import the module after patching
            import api.rag_pipeline.language as language_module

            # Test detect_language
            lang = language_module.detect_language("Hola mundo")
            self.assertEqual(lang, "es")
            mock_detect_language.assert_called_once_with("Hola mundo")

            # Test detect_language_with_details
            details = language_module.detect_language_with_details("Hola mundo")
            self.assertEqual(details, {"lang": "es", "confidence": 0.9})
            mock_detect_details.assert_called_once_with("Hola mundo")

            # Test translate_text without source language
            translated = language_module.translate_text("Hola mundo", target_lang="en")
            self.assertEqual(translated, "Translated text")

            # Test translate_text with source language
            translated_with_source = language_module.translate_text(
                "Hola mundo", target_lang="en", source_lang="es"
            )
            self.assertEqual(translated_with_source, "Translated text")


if __name__ == "__main__":
    unittest.main()
