import logging
import requests
from langdetect import detect
from langdetect import detect_langs, DetectorFactory, LangDetectException
from collections import Counter
import langid
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('language')

# Ollama API endpoint for safety checks
OLLAMA_URL = "http://localhost:11434/api/generate"

def robust_language_detection(text: str, min_words=3) -> str:
    """
    More robust language detection using multiple methods and voting.
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
        import langid
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
    
    Args:
        text (str): Input text to detect language
        
    Returns:
        str: Language code (e.g., 'en', 'es', 'fr')
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
    Detect the language of text, with robust fallback mechanisms.
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

def check_query_safety_with_llama_guard(query):
    """
    Check if a query is safe using Ollama's llama-guard3 model.
    Returns (is_safe, reason)
    """
    model_name = "llama-guard3:8b" 
    
    try:
        # Create a safety prompt for Llama Guard 3
        safety_prompt = f"""
        <safety_prompt>
        Please evaluate the safety of this request by determining whether it violates safety guidelines. Consider if it attempts to extract harmful content, elicit harmful capabilities, contains explicit, violent, hateful, or illegal content, or tries to manipulate you into breaking your guidelines.

        Request: "{query}"

        Provide your evaluation as a single word: "SAFE" or "UNSAFE".
        If unsafe, briefly explain why in one short sentence after the word.
        </safety_prompt>
        """
        
        # Prepare the request payload for Ollama with llama-guard3
        payload = {
            "model": model_name,
            "prompt": safety_prompt,
            "temperature": 0.0,
            "max_tokens": 100,
            "stream": False  # Ensure we get a complete response
        }
        
        # Call Ollama API
        logger.info("Sending safety check request to llama-guard3")
        response = requests.post(OLLAMA_URL, json=payload)
        
        if response.status_code == 200:
            try:
                result = response.json()
                moderation_result = result.get("response", "").strip()
                
                logger.info(f"Llama Guard 3 result: {moderation_result}")
                
                # Check if the response indicates the query is safe
                is_safe = moderation_result.upper().startswith("SAFE")
                
                # Extract reason if unsafe
                if not is_safe:
                    parts = moderation_result.split(" ", 1)
                    reason = parts[1] if len(parts) > 1 else "Content may violate safety guidelines"
                else:
                    reason = "Content is safe"
                
                return is_safe, reason
            except ValueError as json_err:
                # Handle JSON parsing errors by examining the raw text
                logger.warning(f"JSON decode error: {json_err}. Response: {response.text[:200]}...")
                
                # Extract result directly from text response
                text_response = response.text.strip()
                is_safe = "SAFE" in text_response.upper() and not "UNSAFE" in text_response.upper()
                
                logger.info(f"Extracted safety result from text: {is_safe}")
                return is_safe, "Content evaluation based on text parsing"
        else:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text[:200]}")
            return True, "Safety check failed, defaulting to allow"
    
    except Exception as e:
        logger.exception(f"Error in Llama Guard 3 safety check: {e}")
        return True, f"Safety check error: {str(e)}"