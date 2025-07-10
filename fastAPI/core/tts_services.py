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
    Auto-detect language from text with better Hindi/Marathi distinction
    Returns language code (e.g., 'en', 'hi', 'mr')
    """
    if not TTS_AVAILABLE:
        return 'en'
    try:
        # Devanagari script detection
        devanagari = re.search(r'[\u0900-\u097F]', text)
        if devanagari:
            # More distinctive Marathi-only patterns and words
            marathi_patterns = [
                r'\bच्या\b', r'\bत्या\b', r'\bह्या\b',  # Possessive markers unique to Marathi
                r'\bआपल्या\b', r'\bत्यांच्या\b', r'\bतिच्या\b',
                r'\bअसेल\b', r'\bहोतो\b', r'\bहोते\b', r'\bहोती\b',  # Marathi verb endings
                r'\bझाले\b', r'\bझाला\b', r'\bझाली\b',
                r'\bकेले\b', r'\bकेला\b', r'\bकेली\b',
                r'\bआलो\b', r'\bआली\b', r'\bआले\b',
                r'\bगेलो\b', r'\bगेली\b', r'\bगेले\b',
                r'\bपाहिजे\b', r'\bनको\b', r'\bलागते\b',
                r'\bसांगितले\b', r'\bबोललो\b', r'\bविचारले\b'
            ]
            
            # Hindi-specific patterns and words
            hindi_patterns = [
                r'\bहैं\b', r'\bहै\b', r'\bथा\b', r'\bथी\b', r'\bथे\b',  # Hindi copula
                r'\bकरता\s+है\b', r'\bकरती\s+है\b', r'\bकरते\s+हैं\b',
                r'\bहोता\s+है\b', r'\bहोती\s+है\b', r'\bहोते\s+हैं\b',
                r'\bचाहिए\b', r'\bसकता\b', r'\bसकती\b', r'\bसकते\b',
                r'\bरहा\s+है\b', r'\bरही\s+है\b', r'\bरहे\s+हैं\b',
                r'\bगया\b', r'\bगई\b', r'\bगए\b',
                r'\bआया\b', r'\bआई\b', r'\bआए\b',
                r'\bकिया\b', r'\bकी\b', r'\bकिए\b',
                r'\bलिए\b', r'\bद्वारा\b', r'\bअथवा\b',
                r'\bतथा\b', r'\bएवं\b', r'\bहेतु\b'
            ]
            
            # Count matches for each language
            marathi_score = sum(1 for pattern in marathi_patterns if re.search(pattern, text))
            hindi_score = sum(1 for pattern in hindi_patterns if re.search(pattern, text))
            
            # Check for common words that exist in both but with different usage patterns
            # Words like 'आहे' in Marathi vs 'है' in Hindi
            if 'आहे' in text and 'है' not in text:
                marathi_score += 2
            elif 'है' in text or 'हैं' in text:
                hindi_score += 2
            
            # Domain-specific vocabulary hints
            # Government/official Hindi terms
            if any(word in text for word in ['सरकार', 'योजना', 'मंत्रालय', 'विभाग', 'अधिसूचना', 'दिनांक']):
                hindi_score += 1
            
            # Common Marathi daily-use words
            if any(word in text for word in ['बरं', 'नक्की', 'खरंच', 'कधी', 'कुठे', 'कसं']):
                marathi_score += 1
            
            # If scores are very close, check sentence endings
            if abs(marathi_score - hindi_score) <= 1:
                # Marathi often ends with आहे, आहेत
                if re.search(r'आहे[।\s]*$', text) or re.search(r'आहेत[।\s]*$', text):
                    marathi_score += 1
                # Hindi often ends with है, हैं
                elif re.search(r'है[।\s]*$', text) or re.search(r'हैं[।\s]*$', text):
                    hindi_score += 1
            
            # Return based on scores
            if marathi_score > hindi_score:
                return 'mr'
            else:
                return 'hi'  # Default to Hindi for Devanagari text
                
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