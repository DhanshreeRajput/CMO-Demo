import tempfile
import os
import langid
import google.generativeai as genai

# Language detection setup
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not installed. Using langid only.")
    LANGDETECT_AVAILABLE = False

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

def validate_language(text):
    """Check if text is primarily in a supported language."""
    try:
        lang, confidence = langid.classify(text)
        print(f"🗣️ Language detection: {lang} (confidence: {confidence:.2f})")
        return lang in SUPPORTED_LANGUAGES
    except Exception as e:
        print(f"❌ Language validation error: {e}")
        return True

def detect_language_comprehensive(text):
    """
    Comprehensive language detection using multiple methods
    Returns language code with higher accuracy
    """
    try:
        # Clean text for better detection
        clean_text = text.strip()
        if len(clean_text) < 3:
            return 'en'  # Default for very short text
        
        detected_lang = 'en'  # Default
        
        # Method 1: Use langdetect if available (more accurate for Indian languages)
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(clean_text)
                print(f"🔍 Langdetect result: {detected_lang}")
            except:
                pass
        
        # Method 2: Use langid as fallback
        try:
            langid_result, confidence = langid.classify(clean_text)
            print(f"🔍 Langid result: {langid_result} (confidence: {confidence:.2f})")
            
            # If langdetect failed or gave uncertain result, use langid
            if not LANGDETECT_AVAILABLE or confidence > 0.8:
                detected_lang = langid_result
        except:
            pass
        
        # Method 3: Check for specific script patterns
        # Devanagari script detection for Hindi/Marathi
        devanagari_chars = sum(1 for char in clean_text if '\u0900' <= char <= '\u097F')
        total_chars = len([char for char in clean_text if char.isalpha()])
        
        if total_chars > 0:
            devanagari_ratio = devanagari_chars / total_chars
            print(f"🔍 Devanagari ratio: {devanagari_ratio:.2f}")
            
            if devanagari_ratio > 0.3:  # More than 30% Devanagari characters
                # Try to distinguish between Hindi and Marathi
                marathi_words = ['मी', 'तुम्ही', 'आहे', 'आहेत', 'करतो', 'करते', 'म्हणजे', 'पण', 'किंवा', 'अशी', 'तशी']
                hindi_words = ['मैं', 'तुम', 'है', 'हैं', 'करता', 'करते', 'यानी', 'लेकिन', 'या', 'ऐसा', 'वैसा']
                
                marathi_count = sum(1 for word in marathi_words if word in clean_text)
                hindi_count = sum(1 for word in hindi_words if word in clean_text)
                
                if marathi_count > hindi_count:
                    detected_lang = 'mr'
                    print(f"🔍 Script analysis: Detected Marathi (marathi_words: {marathi_count})")
                elif hindi_count > 0:
                    detected_lang = 'hi'
                    print(f"🔍 Script analysis: Detected Hindi (hindi_words: {hindi_count})")
                else:
                    # Default to Hindi for Devanagari if unsure
                    detected_lang = 'hi'
                    print(f"🔍 Script analysis: Defaulting to Hindi for Devanagari script")
        
        # Map to supported languages
        lang_mapping = {'hi': 'hi', 'mr': 'mr', 'en': 'en', 'english': 'en', 'hindi': 'hi', 'marathi': 'mr'}
        final_lang = lang_mapping.get(detected_lang, 'en')
        
        print(f"🎯 Final language detection: {final_lang}")
        return final_lang
        
    except Exception as e:
        print(f"❌ Language detection failed: {e}")
        return 'en'  # Safe default

def transcribe_audio_whisper_local(audio_bytes):
    """
    Transcribe audio using local Whisper model with language detection
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        print(f"🎤 Transcribing audio with local Whisper: {temp_path}")
        
        # Import whisper only when needed
        try:
            import whisper
        except ImportError:
            return (False, "Whisper not installed. Please run: pip install openai-whisper")
        
        # Load Whisper model (downloads first time)
        model = whisper.load_model("base")
        
        # Transcribe with language detection
        result = model.transcribe(temp_path, language=None)  # Auto-detect language
        transcription = result["text"]
        detected_language = result.get("language", "en")
        
        print(f"✅ Local Whisper transcription successful: '{transcription[:50]}...'")
        print(f"🌐 Whisper detected language: {detected_language}")

        # Additional language validation using our comprehensive method
        final_language = detect_language_comprehensive(transcription)
        
        # Validate if language is supported
        if final_language not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            error_msg = (
                f"Sorry, I only support {supported_langs}. "
                "Please speak in one of these languages."
            )
            print(f"❌ Language validation failed: {error_msg}")
            return (False, error_msg)

        return (True, transcription)
        
    except Exception as e:
        error_msg = f"Local Whisper transcription failed: {str(e)}"
        print(f"❌ {error_msg}")
        return (False, error_msg)
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup temporary file: {cleanup_error}")

def transcribe_audio_gemini_api(audio_bytes, google_api_key):
    """
    Transcribe audio using Google Gemini (if supported in future)
    Currently returns fallback message
    """
    return (False, "Google Gemini audio transcription not yet available. Please use text messages or install Whisper for audio support.")

def transcribe_audio_fallback(audio_bytes):
    """
    Fallback transcription method
    """
    return (False, "Audio transcription not available. Please send your message as text in English, Hindi (हिंदी), or Marathi (मराठी).")

def transcribe_audio(client_or_key, audio_bytes, method="local"):
    """
    Transcribe audio using specified method with multi-language support
    Args:
        client_or_key: Google API key or other client
        audio_bytes: Audio file content as bytes
        method: "local", "gemini_api", or "fallback"
    Returns:
        tuple: (success: bool, result: str)
    """
    
    if method == "local":
        return transcribe_audio_whisper_local(audio_bytes)
    elif method == "gemini_api":
        return transcribe_audio_gemini_api(audio_bytes, client_or_key)
    elif method == "fallback":
        return transcribe_audio_fallback(audio_bytes)
    else:
        return (False, f"Unknown transcription method: {method}")

def get_supported_languages():
    """Return dictionary of supported languages"""
    return SUPPORTED_LANGUAGES.copy()

def detect_audio_language(transcription_text):
    """
    Detect language of transcribed text with improved accuracy
    Args:
        transcription_text: The transcribed text
    Returns:
        tuple: (language_code, language_name, confidence)
    """
    try:
        # Use comprehensive language detection
        lang_code = detect_language_comprehensive(transcription_text)
        language_name = SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
        
        # Calculate confidence based on detection methods
        confidence = 0.9 if lang_code in SUPPORTED_LANGUAGES else 0.1
        
        return (lang_code, language_name, confidence)
        
    except Exception as e:
        print(f"❌ Language detection error: {e}")
        return ("en", "English", 0.0)

def is_audio_supported_language(transcription_text):
    """
    Check if transcribed audio is in a supported language
    Args:
        transcription_text: The transcribed text
    Returns:
        bool: True if supported, False otherwise
    """
    lang_code, _, confidence = detect_audio_language(transcription_text)
    
    # Require minimum confidence for language detection
    min_confidence = 0.5
    
    return (
        lang_code in SUPPORTED_LANGUAGES and 
        confidence >= min_confidence
    )

def get_language_specific_response(language, message_type="greeting"):
    """
    Get language-specific responses for different message types
    Args:
        language: Language code ('en', 'hi', 'mr')
        message_type: Type of message ('greeting', 'audio_not_supported', 'help')
    Returns:
        str: Localized response
    """
    
    responses = {
        "greeting": {
            "en": "Hello! I can help you with government schemes and programs. You can send messages in English, Hindi, or Marathi.",
            "hi": "नमस्ते! मैं आपको सरकारी योजनाओं और कार्यक्रमों के बारे में जानकारी दे सकती हूँ। आप अंग्रेजी, हिंदी या मराठी में संदेश भेज सकते हैं।",
            "mr": "नमस्कार! मी तुम्हाला सरकारी योजना आणि कार्यक्रमांबद्दल माहिती देऊ शकते. तुम्ही इंग्रजी, हिंदी किंवा मराठीत संदेश पाठवू शकता."
        },
        "audio_not_supported": {
            "en": "I currently support text messages only. Please send your question as text in English, Hindi, or Marathi.",
            "hi": "मैं फिलहाल केवल टेक्स्ट संदेशों का समर्थन करती हूँ। कृपया अपना प्रश्न अंग्रेजी, हिंदी या मराठी में टेक्स्ट के रूप में भेजें।",
            "mr": "मी सध्या फक्त मजकूर संदेशांना समर्थन देते. कृपया तुमचा प्रश्न इंग्रजी, हिंदी किंवा मराठीत मजकूर म्हणून पाठवा."
        },
        "help": {
            "en": "I can help you with:\n• Government schemes information\n• Eligibility criteria\n• Application processes\n• Benefits and details\n\nSend your questions in English, Hindi (हिंदी), or Marathi (मराठी).",
            "hi": "मैं आपकी इनमें मदद कर सकती हूँ:\n• सरकारी योजनाओं की जानकारी\n• पात्रता मानदंड\n• आवेदन प्रक्रिया\n• लाभ और विवरण\n\nअपने प्रश्न अंग्रेजी, हिंदी या मराठी में भेजें।",
            "mr": "मी तुम्हाला यामध्ये मदत करू शकते:\n• सरकारी योजनांची माहिती\n• पात्रता निकष\n• अर्ज प्रक्रिया\n• फायदे आणि तपशील\n\nतुमचे प्रश्न इंग्रजी, हिंदी किंवा मराठीत पाठवा."
        },
        "language_not_supported": {
            "en": "I support English, Hindi (हिंदी), and Marathi (मराठी) only. Please send your message in one of these languages.",
            "hi": "मैं केवल अंग्रेजी, हिंदी और मराठी का समर्थन करती हूँ। कृपया इनमें से किसी भाषा में अपना संदेश भेजें।",
            "mr": "मी फक्त इंग्रजी, हिंदी आणि मराठी भाषांना समर्थन देते. कृपया यापैकी कोणत्याही भाषेत तुमचा संदेश पाठवा."
        }
    }
    
    return responses.get(message_type, {}).get(language, responses[message_type]["en"])

# Test function for language detection
def test_language_detection():
    """Test function to verify language detection works properly"""
    test_cases = [
        ("Hello, how are you?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("नमस्कार, तुम्ही कसे आहात?", "mr"),
        ("मैं सरकारी योजना के बारे में जानना चाहता हूँ", "hi"),
        ("मला सरकारी योजनेबद्दल माहिती हवी आहे", "mr"),
        ("What government schemes are available?", "en")
    ]
    
    print("🧪 Testing language detection:")
    for text, expected in test_cases:
        detected = detect_language_comprehensive(text)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{text[:30]}...' -> Expected: {expected}, Got: {detected}")

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_language_detection()