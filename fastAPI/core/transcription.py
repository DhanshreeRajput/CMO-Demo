# core/transcription.py - FINAL SOLUTION: Pure Groq API (No PyTorch)
import tempfile
import os
import re

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# Accept if any supported language script is present in the text
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")  # Hindi/Marathi
ENGLISH_REGEX = re.compile(r"[A-Za-z]")

def validate_language(text):
    """Allow only English, Hindi, and Marathi text."""
    if not text or not text.strip():
        return False
    has_devanagari = bool(DEVANAGARI_REGEX.search(text))
    has_english = bool(ENGLISH_REGEX.search(text))
    return has_devanagari or has_english

def is_marathi_text(text):
    """
    Determine if Devanagari text is likely Marathi vs Hindi
    Based on your Streamlit's language detection patterns
    """
    if not text:
        return False
    
    # Strong Marathi-only patterns (these words/patterns are unique to Marathi)
    marathi_unique = [
        'च्या', 'त्या', 'ह्या',  # Possessive forms unique to Marathi
        'आहे', 'आहेत',           # Marathi copula (vs Hindi है/हैं)
        'असेल', 'होतो', 'होते', 'होती',  # Marathi verb forms
        'झाले', 'झाला', 'झाली',  # Marathi past tense
        'केले', 'केला', 'केली',  # Marathi perfect tense
        'पाहिजे', 'लागते', 'नको',  # Marathi modal verbs
        'सांगा', 'विचारा', 'घ्या', 'द्या', 'करा', 'बघा',  # Marathi imperatives
        'येते', 'जाते', 'आलो', 'आली', 'आले', 'गेलो', 'गेली', 'गेले'
    ]
    
    # Strong Hindi-only patterns
    hindi_unique = [
        'है', 'हैं', 'था', 'थी', 'थे',  # Hindi copula and past tense
        'चाहिए', 'सकता', 'सकती', 'सकते',  # Hindi modal verbs
        'करता है', 'करती है', 'करते हैं',  # Hindi present continuous
        'होता है', 'होती है', 'होते हैं',  # Hindi habitual
        'रहा है', 'रही है', 'रहे हैं',    # Hindi progressive
        'गया', 'गई', 'गए', 'किया', 'आया',  # Hindi past forms
        'लिए', 'द्वारा', 'तथा', 'एवं'     # Hindi formal words
    ]
    
    # Count pattern matches
    marathi_score = sum(1 for pattern in marathi_unique if pattern in text)
    hindi_score = sum(1 for pattern in hindi_unique if pattern in text)
    
    print(f"Language analysis - Marathi patterns: {marathi_score}, Hindi patterns: {hindi_score}")
    print(f"Text sample: {text[:100]}...")
    
    # Decision logic
    if marathi_score > 0:
        print("→ Strong Marathi indicators found")
        return True
    elif hindi_score > 0:
        print("→ Strong Hindi indicators found")
        return False
    else:
        # No clear indicators but has Devanagari - default to Marathi
        # This matches how your Streamlit probably handles unclear cases
        if DEVANAGARI_REGEX.search(text):
            print("→ Devanagari script detected, defaulting to Marathi")
            return True
        return False

def transcribe_audio(client, audio_bytes, language=None):
    """
    Pure Groq API transcription with smart Marathi detection
    Works exactly like your Streamlit but without PyTorch dependencies
    """
    print(f"Starting Groq transcription, audio size: {len(audio_bytes)}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        # STEP 1: Initial transcription without language forcing
        with open(temp_path, "rb") as file:
            print("Step 1: Initial Groq transcription (auto-detect)")
            
            result = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="text",
                temperature=0.0
                # No language parameter - let Groq auto-detect first
            )
            
        if isinstance(result, tuple):
            initial_transcription = result[0]
        else:
            initial_transcription = result
            
        print(f"Initial transcription: '{initial_transcription}'")
        
        # STEP 2: Analyze if this should be Marathi
        has_devanagari = bool(DEVANAGARI_REGEX.search(initial_transcription))
        
        if has_devanagari:
            should_be_marathi = is_marathi_text(initial_transcription)
            
            if should_be_marathi:
                print("Step 2: Re-transcribing with Marathi language")
                
                # Re-transcribe forcing Marathi
                with open(temp_path, "rb") as file:
                    result = client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3",
                        response_format="text",
                        temperature=0.0,
                        language="mr"  # Force Marathi
                    )
                    
                if isinstance(result, tuple):
                    final_transcription = result[0]
                else:
                    final_transcription = result
                    
                print(f"Marathi transcription: '{final_transcription}'")
                detected_lang = 'mr'
                confidence = 0.85
            else:
                print("Step 2: Re-transcribing with Hindi language")
                
                # Re-transcribe forcing Hindi
                with open(temp_path, "rb") as file:
                    result = client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3",
                        response_format="text",
                        temperature=0.0,
                        language="hi"  # Force Hindi
                    )
                    
                if isinstance(result, tuple):
                    final_transcription = result[0]
                else:
                    final_transcription = result
                    
                print(f"Hindi transcription: '{final_transcription}'")
                detected_lang = 'hi'
                confidence = 0.85
        else:
            # English or other - use initial transcription
            final_transcription = initial_transcription
            detected_lang = 'en'
            confidence = 0.9
            print("Using initial transcription (English)")

        # STEP 3: Validation
        if detected_lang not in ["hi", "mr", "en"] or confidence < 0.7:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            return (
                False,
                f"Sorry, I only support {supported_langs}. Please speak in one of these languages. (Detected: {detected_lang}, confidence: {confidence:.2f})"
            )
        
        if not validate_language(final_transcription):
            print(f"Final validation failed for: '{final_transcription}'")
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            return (
                False,
                f"Sorry, I only support {supported_langs}. Please speak in one of these languages."
            )
        
        print(f"✅ Transcription successful: '{final_transcription}' (Language: {detected_lang})")
        # Remove . , - and " characters from the transcription
        cleaned_transcription = re.sub(r'[.,\-\"]', '', final_transcription)
        # Remove leading/trailing spaces and collapse multiple spaces
        cleaned_transcription = re.sub(r'\s+', ' ', cleaned_transcription).strip()
        # Join sequences of single uppercase letters (acronyms)
        def join_acronyms(text):
            return re.sub(r'(?<!\w)((?:[A-Z]\s+){1,}[A-Z])(?!\w)', lambda m: m.group(0).replace(' ', ''), text)
        cleaned_transcription = join_acronyms(cleaned_transcription)
        return (True, cleaned_transcription)
        
    except Exception as e:
        print(f"❌ Transcription exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)