import re
from io import BytesIO
import time

# Import cache functions/objects directly
from core.cache_manager import (
    _audio_cache,
    get_audio_hash,
    cache_audio,
    get_cached_audio
)

# New imports for TTS and language detection
try:
    from gtts import gTTS
    import pygame
    from langdetect import detect, DetectorFactory
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS dependencies not installed. Install with: pip install gtts pygame langdetect")

def detect_language(text):
    """
    Auto-detect language from text
    Returns language code (e.g., 'en', 'hi', 'mr')
    """
    if not TTS_AVAILABLE:
        return 'en'
    try:
        # Devanagari script detection
        devanagari = re.search(r'[\u0900-\u097F]', text)
        if devanagari:
            # Marathi keyword check (add more as needed)
            marathi_keywords = ['च्या', 'आहे', 'नाही', 'होय', 'माझ्या', 'तुमच्या', 'आपल्या', 'कृपया', 'कारण', 'म्हणून']
            if any(word in text for word in marathi_keywords):
                return 'mr'
            return 'hi'
        # English script
        if re.search(r'[A-Za-z]', text):
            return 'en'
        return 'en'
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'  # Fallback to English

def text_to_speech(text, lang=None, auto_detect=True, speed=1.0):
    """
    Convert text to speech with caching and speed control.
    Returns: (audio_bytes, language_used, cache_status)
    """
    if not TTS_AVAILABLE:
        return None, 'en', 'TTS not available'
    
    try:
        # Auto-detect language if not provided
        if auto_detect or not lang:
            detected_lang = detect_language(text)
            lang = detected_lang

        # Only allow TTS for en, hi, mr
        if lang not in {'en', 'hi', 'mr'}:
            raise ValueError(f"TTS only supports English, Hindi, and Marathi. Detected: {lang}")

        # Use cache directly
        audio_hash = get_audio_hash(text, lang, speed)
        cached_audio = get_cached_audio(audio_hash)
        if cached_audio:
            return cached_audio, lang, 'cached'

        # Use gTTS with explicit language agent
        if lang == 'en':
            tts = gTTS(text=text, lang='en', slow=speed < 0.8)
        elif lang == 'hi':
            tts = gTTS(text=text, lang='hi', slow=speed < 0.8)
        elif lang == 'mr':
            tts = gTTS(text=text, lang='mr', slow=speed < 0.8)
        else:
            raise ValueError(f"Unsupported language: {lang}")

        # Save to BytesIO buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.getvalue()

        cache_audio(audio_hash, audio_bytes)
        return audio_bytes, lang, 'generated'
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None, lang or 'en', f'error: {str(e)}'

def generate_audio_response(text, lang_preference=None, speed=1.0):
    """
    Generate audio response for given text.
    Returns: (audio_data, detected_lang, cache_hit) tuple
    """
    if not TTS_AVAILABLE:
        return None, 'en', False

    try:
        # Clean the input text
        clean_text = re.sub(r'\[.*?\/]', '', text)
        clean_text = re.sub(r'[✅ℹ️🔍⚠️*●#=]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if len(clean_text) < 5: # Min length for meaningful TTS
            return None, (lang_preference if lang_preference != 'auto' else 'en'), False

        target_lang = lang_preference if lang_preference and lang_preference != 'auto' else None
        
        audio_bytes, final_lang, cache_status = text_to_speech(
            clean_text,
            lang=target_lang, # Pass specific lang if chosen, else None for auto-detect
            auto_detect=(not target_lang), # Auto-detect only if no specific lang is preferred
            speed=speed
        )

        cache_hit = cache_status == 'cached'
        
        # If lang_preference was 'auto', final_lang is the detected one.
        # If a specific lang was preferred, final_lang is that preferred one (or fallback if error).
        return audio_bytes, final_lang, cache_hit

    except Exception as e:
        print(f"Error generating audio in generate_audio_response: {str(e)}")
        return None, (lang_preference if lang_preference != 'auto' else 'en'), False