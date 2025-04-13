"""
Language detection and translation functionality for the Ollama RAG system.
"""
from deep_translator import GoogleTranslator
from langdetect import detect
from langdetect import detect_langs, DetectorFactory, LangDetectException
from collections import Counter
import langid
from rag_pipeline.config import logger

def robust_language_detection(text: str, min_words=3) -> str:
    """
    Detect language using multiple methods and majority voting.
    Falls back to sensible defaults for short texts.
    """
    DetectorFactory.seed = 0

    text = text.strip()
    if not text:
        return 'en'

    # Short command heuristic
    if len(text.split()) < min_words:
        common_en = {
            'ok', 'yes', 'no', 'quit', 'exit', 'help', 'thanks',
            'hi', 'hello', 'start', 'stop', 'please'
        }
        if text.lower() in common_en:
            logger.info(f"Detected '{text}' as common English command")
            return 'en'

    langs = []

    # langdetect vote-based
    try:
        segment_len = max(len(text) // 5, 20)
        segments = [text[i:i+segment_len] for i in range(0, len(text), segment_len)]
        langs += [detect(s) for s in segments if len(s.strip()) >= 10]
        logger.info(f"langdetect votes: {langs}")
    except LangDetectException:
        pass

    # langid fallback
    try:
        langid_result, confidence = langid.classify(text)
        langs.append(langid_result)
        logger.info(f"langid detection: {langid_result} (confidence={confidence:.2f})")
    except ImportError:
        logger.warning("langid not installed")

    # polyglot (optional)
    try:
        from polyglot.detect import Detector
        lang_poly = Detector(text).language.code
        langs.append(lang_poly)
        logger.info(f"polyglot detection: {lang_poly}")
    except Exception:
        logger.debug("Polyglot not available or failed.")

    # Majority vote
    if langs:
        vote = Counter(langs).most_common(1)[0][0]
        logger.info(f"Final language: {vote} (votes={Counter(langs)})")

        # Extra caution for short inputs
        if len(text.split()) < min_words and vote != 'en' and Counter(langs)[vote] < 2:
            logger.info(f"Low vote count for short input, defaulting to 'en'")
            return 'en'

        return vote

    logger.warning("All detection methods failed. Defaulting to English.")
    return 'en'

def simplified_language_detection(text):
    """
    A simplified language detection function with fewer dependencies.
    Falls back to simple heuristics for short texts.
    """
    # For very short queries, default to English to avoid misclassification
    if len(text.split()) < 3:
        logger.info(f"Text '{text}' is too short for reliable detection, defaulting to English")
        return 'en'
    
    # Common English command check
    common_english = {
        'quit', 'exit', 'help', 'stop', 'hi', 'hello', 'thanks', 'yes', 'no',
        'please', 'show', 'find', 'search', 'get', 'list'
    }
    if text.lower() in common_english:
        return 'en'
    
    # Try langdetect as primary method
    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        logger.warning(f"Language detection error: {e}. Defaulting to English.")
        return 'en'

def detect_language(text: str) -> str:
    """
    Detect language of input text, with fallback mechanisms.
    """
    try:
        return robust_language_detection(text)
    except Exception as e:
        logger.warning(f"Robust detection failed: {e}, falling back...")
        return simplified_language_detection(text)

def translate_text(text, target_lang='en', source_lang=None):
    """
    Translate text using deep-translator.
    """
    try:
        if not text:
            return text
        
        source = source_lang if source_lang else 'auto'
        translator = GoogleTranslator(source=source, target=target_lang)
        result = translator.translate(text)
        return result
    except Exception as e:
        logger.warning(f"Translation error: {e}. Returning original text.")
        return text