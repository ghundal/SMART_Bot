"""
Language detection module for the Ollama RAG pipeline.
Combines FastText with pattern matching for efficient, accurate language detection.
"""

import re
from typing import Dict, Optional, Union, cast

from deep_translator import GoogleTranslator

# Required external dependencies
from fast_langdetect import detect as fasttext_detect

# Import from rag_pipeline
from rag_pipeline.config import logger


class LanguageDetector:
    """
    A hybrid language detector that combines pattern matching for short commands
    with FastText for general text.
    """

    def __init__(self, default_language: str = "en", min_confidence: float = 0.5):
        """
        Initialize the language detector.

        Args:
            default_language: Default language to return if detection fails
            min_confidence: Minimum confidence threshold for FastText detection
        """
        self.default_language = default_language
        self.min_confidence = min_confidence

        # Define common patterns for various languages
        self.patterns = {
            # English patterns
            "en": [
                # Question patterns
                r"^(what|who|when|where|why|how)(\s|\'s|\'re|$)",
                # Command patterns
                r"^(tell|show|give|find|search|look|get|fetch|list|summarize|explain|describe)(\s|$)",
                # Modal verb patterns
                r"^(can|could|would|should|may|might|must)(\s|$)",
                # Be verb patterns
                r"^(is|are|was|were|be|been|am)(\s|$)",
                # Do verb patterns
                r"^(do|does|did|don\'t|doesn\'t|didn\'t)(\s|$)",
                # Common starts
                r"^(i|we|you|they|he|she|it)(\s|$)",
                # Polite requests
                r"^(please|kindly)(\s|$)",
                # Common short commands
                r"^(ok|yes|no|help|exit|quit|stop|start)$",
                # Common greetings
                r"^(hi|hello|hey|bye|goodbye)(\s|$)",
                # Assumptions case
                r"^(assumptions|assumptions?)(\s|$)",
            ],
            # Spanish patterns
            "es": [
                r"^(qué|quien|cuándo|dónde|por qué|cómo)(\s|$)",
                r"^(dime|muéstrame|búscame|encuentre)(\s|$)",
                r"^(hola|adiós|gracias|por favor)(\s|$)",
                r"^(sí|no|ayuda|salir|parar)(\s|$)",
            ],
            # French patterns
            "fr": [
                r"^(qui|quoi|quand|où|pourquoi|comment)(\s|$)",
                r"^(dis-moi|montre-moi|trouve-moi)(\s|$)",
                r"^(bonjour|salut|au revoir|merci)(\s|$)",
                r"^(oui|non|aide|quitter|arrêter)(\s|$)",
            ],
            # German patterns
            "de": [
                r"^(wer|was|wann|wo|warum|wie)(\s|$)",
                r"^(sag mir|zeig mir|finde)(\s|$)",
                r"^(hallo|tschüss|danke|bitte)(\s|$)",
                r"^(ja|nein|hilfe|beenden|stopp)(\s|$)",
            ],
            # Add more languages as needed
        }

        # Common English phrases that are often misclassified
        self.common_english_phrases = {
            "tell me",
            "show me",
            "give me",
            "help me",
            "can you",
            "could you",
            "would you",
            "should you",
            "tell me assumptions",
            "show me data",
            "assumptions",
            "find information",
            "search for",
            "look up",
            "what is",
            "how to",
            "why is",
            "when will",
            "who is",
            "where is",
            "which one",
            "whose",
        }

        # Initialize the translator for potential future use
        self.translator = GoogleTranslator(source="auto", target="en")

        logger.debug("LanguageDetector initialized successfully")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text using a hybrid approach.

        Args:
            text: Text to detect language for

        Returns:
            ISO 639-1 language code (2 letters)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning default language")
            return self.default_language

        text = text.strip()
        word_count = len(text.split())

        logger.debug(f"Detecting language for: '{text}' (word count: {word_count})")

        # Special handling for very short texts (1-3 words)
        if word_count <= 3:
            # Check for common English phrases
            text_lower = text.lower()
            for phrase in self.common_english_phrases:
                if text_lower == phrase or text_lower.startswith(phrase + " "):
                    logger.debug(f"Matched common English phrase: '{phrase}'")
                    return "en"

            # Check for language-specific patterns
            pattern_result = self._check_patterns(text_lower)
            if pattern_result:
                logger.debug(f"Matched pattern for language: {pattern_result}")
                return pattern_result

        # Use FastText for general language detection
        try:
            result = fasttext_detect(
                text, low_memory=False
            )  # Use the full model for better accuracy
            lang = cast(str, result["lang"])
            confidence = result["score"]

            logger.debug(f"FastText detection: {lang} (confidence: {confidence:.4f})")

            # If confidence is too low and it's a short text, apply bias toward English
            if confidence < self.min_confidence and word_count <= 3:
                logger.debug(
                    f"Low confidence ({confidence:.4f}) for short text, defaulting to English"
                )
                return "en"

            return lang

        except Exception as e:
            logger.error(f"Error in FastText detection: {e}")

            # Fallback to pattern matching
            pattern_result = self._check_patterns(text.lower())
            if pattern_result:
                logger.debug(f"Fallback pattern match: {pattern_result}")
                return pattern_result

            logger.warning(
                f"All detection methods failed, returning default: {self.default_language}"
            )
            return self.default_language

    def _check_patterns(self, text: str) -> Optional[str]:
        """
        Check if the text matches any language-specific patterns.

        Args:
            text: Text to check (should be lowercase)

        Returns:
            Language code if matched, None otherwise
        """
        for lang, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return lang
        return None

    def detect_with_details(self, text: str) -> Dict[str, Union[str, float, int]]:
        """
        Detect language with detailed information.

        Args:
            text: Text to detect language for

        Returns:
            Dictionary with detection details
        """
        if not text or not text.strip():
            return {
                "lang": self.default_language,
                "method": "default",
                "confidence": 0.0,
                "word_count": 0,
            }

        text = text.strip()
        word_count = len(text.split())
        text_lower = text.lower()

        # Check for common English phrases
        if word_count <= 3:
            for phrase in self.common_english_phrases:
                if text_lower == phrase or text_lower.startswith(phrase + " "):
                    return {
                        "lang": "en",
                        "method": f"common_phrase:{phrase}",
                        "confidence": 0.95,
                        "word_count": word_count,
                    }

        # Check for language-specific patterns
        for lang, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return {
                        "lang": lang,
                        "method": f"pattern:{pattern}",
                        "confidence": 0.9,
                        "word_count": word_count,
                    }

        # Use FastText for general language detection
        try:
            result = fasttext_detect(text, low_memory=False)
            return {
                "lang": cast(str, result["lang"]),
                "method": "fasttext",
                "confidence": result["score"],
                "word_count": word_count,
            }
        except Exception as e:
            logger.error(f"FastText detection error: {e}")
            return {
                "lang": self.default_language,
                "method": "default_fallback",
                "confidence": 0.0,
                "word_count": word_count,
                "error": str(e),
            }

    def translate_text(self, text: str, target_lang: str = "en") -> str:
        """
        Translate text to the target language.

        Args:
            text: Text to translate
            target_lang: Target language code

        Returns:
            Translated text
        """
        if not text:
            return ""

        try:
            source_lang = self.detect_language(text)
            if source_lang == target_lang:
                return text

            self.translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = self.translator.translate(text)
            if translated is None:
                return ""
            return cast(str, translated)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""


# Singleton instance for reuse
_detector = None


def get_detector() -> LanguageDetector:
    """
    Get or create the singleton detector instance.

    Returns:
        LanguageDetector instance
    """
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector


def detect_language(text: str) -> str:
    """
    Detect language using the shared detector instance.

    Args:
        text: Text to detect language for

    Returns:
        ISO 639-1 language code
    """
    detector = get_detector()
    return detector.detect_language(text)


def detect_language_with_details(text: str) -> Dict[str, Union[str, float, int]]:
    """
    Detect language with detailed information about the detection process.

    Args:
        text: Text to detect language for

    Returns:
        Dictionary with language detection details
    """
    detector = get_detector()
    return detector.detect_with_details(text)


def translate_text(text: str, target_lang: str = "en", source_lang: Optional[str] = None) -> str:
    """
    Translate text to the target language.

    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code (optional)

    Returns:
        Translated text
    """
    if not text:
        return ""

    try:
        detector = get_detector()

        # If source language is provided, use it; otherwise detect it
        if not source_lang:
            source_lang = detector.detect_language(text)

        # If source and target are the same, return original text
        if source_lang == target_lang:
            return text

        # Initialize translator with the appropriate source language
        detector.translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = detector.translator.translate(text)
        if translated is None:
            return ""
        return cast(str, translated)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return ""
