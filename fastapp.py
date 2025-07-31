from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import json
import redis
from contextlib import asynccontextmanager
import os
import logging
import re
import requests
import PyPDF2
import traceback
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import aiohttp
import asyncio
from urllib.parse import urlparse
import sys

# Fix Windows Unicode encoding issues
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    except:
        pass

# Safe logging for Windows
class SafeFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if sys.platform.startswith('win'):
            emoji_replacements = {
                '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🔍': '[SEARCH]',
                '🤖': '[AI]', '📤': '[SEND]', '📝': '[MSG]', '🔄': '[RETRY]',
                '💬': '[CHAT]', '⏱️': '[RATE_LIMIT]', '⚡': '[FAST]', '🚀': '[SPEED]'
            }
            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
        return msg

def setup_safe_logging():
    formatter = SafeFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler('whatsapp.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

setup_safe_logging()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# OPTIMIZED ENVIRONMENT VARIABLES FOR SPEED
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")  # Default to fast model
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# SPEED OPTIMIZATION SETTINGS - UPDATED FOR FULL RESPONSES
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"
MAX_RESPONSE_TIME = int(os.getenv("MAX_RESPONSE_TIME", 60))  # Increased for full responses
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))
CACHE_TTL = int(os.getenv("CACHE_TTL", 1800))

# ENHANCED DEVANAGARI LANGUAGE DETECTION
def detect_language(text):
    """Enhanced Devanagari-aware language detection for Marathi, Hindi, and English"""
    try:
        clean_text = text.strip().lower()
        if len(clean_text) < 2:
            return 'en'

        # Check for Devanagari characters
        devanagari_chars = bool(re.search(r'[\u0900-\u097F]', clean_text))
        english_chars = bool(re.search(r'[a-zA-Z]', clean_text))

        if devanagari_chars:
            # ENHANCED MARATHI DETECTION - More comprehensive word lists
            marathi_strong_indicators = [
                # Common Marathi verbs and auxiliaries
                'आहे', 'आहेत', 'होते', 'होती', 'होता', 'असे', 'असा', 'अशी',
                # Marathi pronouns
                'तुम्ही', 'तुमचा', 'तुमची', 'तुमचे', 'तुमच्या', 'मी', 'माझा', 'माझी', 'माझे', 'माझ्या',
                # Marathi specific words
                'माहिती', 'येथे', 'तेथे', 'इथे', 'करा', 'कसा', 'कसे', 'कशी', 'बद्दल', 'द्या', 'घ्या',
                'सांगा', 'सांगू', 'मला', 'तुला', 'त्याला', 'तिला', 'आम्हाला', 'तुमाला',
                # Marathi particles and postpositions
                'ला', 'च्या', 'मध्ये', 'वर', 'खाली', 'समोर', 'मागे', 'शेजारी',
                # Marathi question words
                'काय', 'कोण', 'कुठे', 'केव्हा', 'कसे', 'किती', 'कशासाठी',
                # Common Marathi words
                'पाहिजे', 'हवे', 'नको', 'चालू', 'बंद', 'नवीन', 'जुने', 'मोठे', 'छोटे',
                'चांगले', 'वाईट', 'लवकर', 'उशीर', 'आज', 'उद्या', 'परवा'
            ]
            
            marathi_weak_indicators = [
                # Additional Marathi markers
                'ते', 'तो', 'ती', 'हे', 'हा', 'ही', 'या', 'यो', 'यू', 'त्या', 'त्यो',
                'नाही', 'नसते', 'गेले', 'आले', 'झाले', 'केले', 'दिले', 'घेतले'
            ]
            
            # ENHANCED HINDI DETECTION
            hindi_strong_indicators = [
                # Common Hindi verbs and auxiliaries  
                'है', 'हैं', 'था', 'थी', 'थे', 'होगा', 'होगी', 'होंगे', 'गया', 'गई', 'गए',
                # Hindi pronouns
                'आप', 'आपका', 'आपकी', 'आपके', 'आपको', 'मैं', 'मेरा', 'मेरी', 'मेरे', 'मुझे', 'मुझको',
                # Hindi specific words
                'जानकारी', 'यहाँ', 'वहाँ', 'कहाँ', 'करना', 'कैसे', 'कैसा', 'कैसी', 'के बारे में', 'बताएं', 'बताओ',
                'कहें', 'कहिए', 'मुझे', 'आपको', 'उसे', 'उसको', 'हमें', 'हमको', 'उन्हें',
                # Hindi particles and postpositions
                'को', 'का', 'की', 'के', 'में', 'पर', 'से', 'तक', 'के लिए', 'के साथ',
                # Hindi question words
                'क्या', 'कौन', 'कहाँ', 'कब', 'कैसे', 'कितना', 'कितनी', 'कितने', 'क्यों', 'किसलिए',
                # Common Hindi words
                'चाहिए', 'चाहें', 'मत', 'चालू', 'बंद', 'नया', 'पुराना', 'बड़ा', 'छोटा',
                'अच्छा', 'बुरा', 'जल्दी', 'देर', 'आज', 'कल', 'परसों'
            ]
            
            hindi_weak_indicators = [
                # Additional Hindi markers
                'वह', 'वो', 'यह', 'ये', 'इस', 'उस', 'इन', 'उन', 'वे', 'तुम', 'तू',
                'नहीं', 'मत', 'रहा', 'रही', 'रहे', 'कर', 'किया', 'दिया', 'लिया'
            ]

            # Count strong indicators (weighted more heavily)
            marathi_strong_count = sum(2 for word in marathi_strong_indicators if word in clean_text)
            marathi_weak_count = sum(1 for word in marathi_weak_indicators if word in clean_text)
            marathi_total = marathi_strong_count + marathi_weak_count

            hindi_strong_count = sum(2 for word in hindi_strong_indicators if word in clean_text)
            hindi_weak_count = sum(1 for word in hindi_weak_indicators if word in clean_text)
            hindi_total = hindi_strong_count + hindi_weak_count

            print(f"🔍 Devanagari analysis - Marathi: {marathi_total} (strong: {marathi_strong_count//2}, weak: {marathi_weak_count}), Hindi: {hindi_total} (strong: {hindi_strong_count//2}, weak: {hindi_weak_count})")

            # Decision logic with bias handling
            if marathi_total > hindi_total:
                return 'mr'
            elif hindi_total > marathi_total:
                return 'hi'
            else:
                # Equal scores - check for romanized hints
                marathi_roman = ['baddal', 'mahiti', 'dya', 'kasa', 'kara', 'ahe', 'tumhi', 'mala', 'sangha', 'kay']
                hindi_roman = ['kaise', 'karna', 'batao', 'mujhe', 'aapka', 'kya', 'hai', 'hain']
                
                marathi_roman_count = sum(1 for word in marathi_roman if word in clean_text)
                hindi_roman_count = sum(1 for word in hindi_roman if word in clean_text)
                
                if marathi_roman_count > 0:
                    return 'mr'
                elif hindi_roman_count > 0:
                    return 'hi'
                else:
                    # Final fallback - check for specific character patterns
                    # Marathi tends to use certain conjuncts more frequently
                    marathi_conjuncts = ['ण्', 'ळ', 'झ्', 'भ्र', 'क्ष्', 'ज्ञ्']
                    hindi_patterns = ['त्र्', 'श्र्', 'क्र्', 'प्र्']
                    
                    marathi_pattern_count = sum(1 for pattern in marathi_conjuncts if pattern in clean_text)
                    hindi_pattern_count = sum(1 for pattern in hindi_patterns if pattern in clean_text)
                    
                    if marathi_pattern_count > hindi_pattern_count:
                        return 'mr'
                    else:
                        return 'hi'  # Default to Hindi for ambiguous Devanagari

        elif english_chars and not devanagari_chars:
            # Pure English/Roman script - check for romanized Indian languages
            marathi_roman_words = [
                'baddal', 'mahiti', 'dya', 'kasa', 'kara', 'ahe', 'aahe', 'tumhi', 
                'mala', 'tula', 'sangha', 'sangu', 'kay', 'kuthe', 'kiti', 'kevha',
                'pahije', 'have', 'nako', 'chalu', 'band', 'navin', 'june', 'mothe', 'chote',
                'changale', 'vait', 'lavkar', 'ushir', 'aaj', 'udya', 'parva'
            ]
            
            hindi_roman_words = [
                'kaise', 'kaisa', 'kaisi', 'karna', 'batao', 'bataiye', 'mujhe', 'aapko', 
                'usse', 'hamen', 'unhen', 'kya', 'kaun', 'kahan', 'kab', 'kitna', 'kitni',
                'chahiye', 'chahen', 'mat', 'chalu', 'band', 'naya', 'purana', 'bada', 'chota',
                'accha', 'bura', 'jaldi', 'der', 'aaj', 'kal', 'parson'
            ]
            
            marathi_roman_count = sum(1 for word in marathi_roman_words if word in clean_text)
            hindi_roman_count = sum(1 for word in hindi_roman_words if word in clean_text)
            
            print(f"🔍 Roman script analysis - Marathi: {marathi_roman_count}, Hindi: {hindi_roman_count}")
            
            if marathi_roman_count > 0 and marathi_roman_count >= hindi_roman_count:
                return 'mr'
            elif hindi_roman_count > 0:
                return 'hi'
            else:
                return 'en'  # Default English for pure Roman script

        else:
            return 'en'  # Default fallback
            
    except Exception as e:
        print(f"❌ Language detection error: {e}")
        return 'en'

# FAST RESPONSE CACHE
class FastResponseCache:
    def __init__(self, max_size: int = 300, ttl_seconds: int = 1800):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.stats = {"hits": 0, "misses": 0}
    
    def get_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        key = self.get_key(query)
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data["timestamp"] < self.ttl:
                self.stats["hits"] += 1
                print("💨 CACHE HIT - Instant response!")
                return cached_data["response"]
            else:
                del self.cache[key]  # Remove expired
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, response: str):
        key = self.get_key(query)
        
        # Clear old entries if cache is full
        if len(self.cache) >= self.max_size:
            old_keys = list(self.cache.keys())[:20]
            for old_key in old_keys:
                del self.cache[old_key]
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        print("💾 Response cached for future speed")
    
    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }

# Initialize fast cache
fast_cache = FastResponseCache(max_size=300, ttl_seconds=CACHE_TTL)

# Import services with fallback
try:
    from core.rag_services import (
        detect_language as transcription_detect_language,
        build_rag_chain_from_documents,
        process_scheme_query_with_retry,
        clear_query_cache,
        get_model_options,
        get_cache_stats
    )
    RAG_SERVICES_AVAILABLE = True
    print("✅ RAG services imported successfully")
except ImportError as e:
    print(f"⚠️ rag_services.py import failed: {e}")
    RAG_SERVICES_AVAILABLE = False

# Import transcription with fallback
try:
    from core.transcription import transcribe_audio
    TRANSCRIPTION_AVAILABLE = True
    print("✅ Transcription services imported successfully")
except ImportError:
    print("⚠️ transcription.py not found")
    TRANSCRIPTION_AVAILABLE = False

def handle_response_format(response_text, user_language):
    """Enhanced response formatting with strict language matching"""
    # Normalize language codes
    lang_mapping = {
        'english': 'en',
        'hindi': 'hi', 
        'marathi': 'mr',
        'en': 'en',
        'hi': 'hi',
        'mr': 'mr'
    }
    
    normalized_lang = lang_mapping.get(user_language.lower(), 'en')
    
    helpline_endings = {
        'en': "\n\n**For more details, please contact the 104/102 helpline numbers.**",
        'hi': "\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
        'mr': "\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
    }

    # Remove any existing helpline endings first
    for ending in helpline_endings.values():
        response_text = response_text.replace(ending.strip(), "")
    
    # Add correct helpline ending based on detected language
    correct_ending = helpline_endings.get(normalized_lang, helpline_endings['en'])
    response_text = response_text.strip() + correct_ending

    return response_text

def validate_response_format(response_text, user_language):
    """Quick response validation"""
    return len(response_text) > 20 and ('104' in response_text or '102' in response_text)

def validate_devanagari_response(response_text: str, expected_language: str) -> bool:
    """Validate that Devanagari responses match expected language"""
    try:
        if expected_language not in ['hi', 'mr']:
            return True  # No validation needed for English
        
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', response_text))
        if not has_devanagari:
            return False  # Should have Devanagari for Hindi/Marathi
        
        # Check for language-specific patterns in response
        detected_lang = detect_language(response_text[:200])
        return detected_lang == expected_language
        
    except Exception as e:
        print(f"❌ Devanagari validation error: {e}")
        return False

# ENHANCED GREETING FUNCTION WITH BETTER DEVANAGARI SUPPORT
def get_instant_greeting(query: str, language: str) -> Optional[str]:
    """Get instant greeting responses with enhanced Devanagari handling"""
    query_lower = query.strip().lower()
    
    # Normalize language
    lang_mapping = {
        'english': 'en',
        'hindi': 'hi', 
        'marathi': 'mr',
        'en': 'en',
        'hi': 'hi',
        'mr': 'mr'
    }
    normalized_lang = lang_mapping.get(language.lower(), 'en')
    
    # Enhanced greeting detection including Devanagari
    greetings = [
        'hi', 'hello', 'hey', 'namaste', 'नमस्ते', 'नमस्कार', 
        'good morning', 'good afternoon', 'good evening',
        'हैलो', 'हाय', 'सुप्रभात', 'शुभ संध्या', 'शुभ सकाळ',
        'धन्यवाद', 'thank you', 'thanks', 'आभार'
    ]
    
    if query_lower in greetings or any(greet in query_lower for greet in greetings):
        greetings_responses = {
            'hi': "नमस्ते! मैं आपकी सरकारी योजनाओं संबंधी प्रश्नों में सहायता करती हूँ। आप कैसे पूछना चाहते हैं? 😊\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "नमस्कार! मी सरकारी योजनांविषयी तुमच्या प्रश्नांमध्ये मदत करते. तुम्हाला काय जाणून घ्यायचे आहे? 😊\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "Hello! I help with questions about government schemes and programs. What would you like to know? 😊\n\n**For more details, please contact the 104/102 helpline numbers.**"
        }
        return greetings_responses.get(normalized_lang, greetings_responses['en'])
    
    return None

# ENHANCED WhatsApp message sending with better error handling
def send_whatsapp_message(to: str, message: str) -> bool:
    """Enhanced WhatsApp message sending with network resilience"""
    max_retries = 3
    base_timeout = 15
    
    for attempt in range(max_retries):
        try:
            timeout = base_timeout + (attempt * 5)  # Increase timeout with each retry
            
            headers = {
                'Authorization': f'Bearer {WHATSAPP_TOKEN}',
                'Content-Type': 'application/json',
                'User-Agent': 'WhatsApp-Bot/1.0'
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": message}
            }
            
            url = f"https://graph.facebook.com/v12.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            
            print(f"📤 Sending to WhatsApp (attempt {attempt + 1}/{max_retries}, timeout: {timeout}s)")
            
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=timeout,
                verify=True  # Ensure SSL verification
            )
            
            if response.status_code == 200:
                logging.info(f"[OK] WhatsApp message sent to {to}")
                return True
            elif response.status_code == 401:
                # Token issue - don't retry
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', '')
                except:
                    error_message = response.text
                
                logging.error(f"[TOKEN] WhatsApp token issue: {error_message}")
                return False
            elif response.status_code == 429:
                # Rate limit - wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"[RATE_LIMIT] Waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"[ERROR] WhatsApp API error: {response.status_code}")
                try:
                    error_details = response.json()
                    logging.error(f"   Details: {error_details}")
                except:
                    logging.error(f"   Response: {response.text[:200]}")
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Brief wait before retry
                    continue
                return False
                
        except requests.exceptions.Timeout:
            logging.error(f"[TIMEOUT] WhatsApp API timeout after {timeout}s (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return False
        except requests.exceptions.ConnectionError as e:
            logging.error(f"[CONNECTION] WhatsApp connection error: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 + (attempt * 3)  # Longer wait for connection issues
                print(f"⏳ Waiting {wait_time}s for network recovery...")
                time.sleep(wait_time)
                continue
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"[REQUEST] WhatsApp request error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False
        except Exception as e:
            logging.error(f"[CRASH] Unexpected WhatsApp error: {str(e)}")
            return False
    
    logging.error(f"[FAILED] All {max_retries} WhatsApp send attempts failed")
    return False

# Validation functions
def validate_whatsapp_config():
    """Validate WhatsApp configuration"""
    required_vars = [WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN]
    missing = [var for var in required_vars if not var]
    
    if missing:
        logging.error(f"[ERROR] Missing WhatsApp config: {len(missing)} variables")
        return False
    
    # Quick token test with better error handling
    try:
        headers = {'Authorization': f'Bearer {WHATSAPP_TOKEN}'}
        url = f"https://graph.facebook.com/v12.0/{WHATSAPP_PHONE_NUMBER_ID}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logging.info("[OK] WhatsApp token valid")
            return True
        else:
            logging.error(f"[TOKEN] WhatsApp token issue: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"[ERROR] WhatsApp validation failed: {e}")
        return False

def validate_ollama_server():
    """Validate Ollama server and test speed"""
    try:
        start_time = time.time()
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        connection_time = round((time.time() - start_time) * 1000, 2)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if MODEL_NAME in model_names:
                print(f"✅ Model '{MODEL_NAME}' available ({connection_time}ms)")
                
                # Test generation speed
                test_start = time.time()
                test_payload = {
                    "model": MODEL_NAME,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 5}
                }
                
                gen_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate", 
                    json=test_payload, 
                    timeout=15
                )
                gen_time = round((time.time() - test_start) * 1000, 2)
                
                if gen_response.status_code == 200:
                    print(f"⚡ Speed test: {gen_time}ms")
                    if gen_time < 3000:
                        print("🔥 EXCELLENT speed for WhatsApp!")
                    elif gen_time < 8000:
                        print("✅ Good speed for WhatsApp")
                    else:
                        print("⚠️ May be slow for WhatsApp")
                    return True
                
            else:
                print(f"❌ Model '{MODEL_NAME}' not found")
                print(f"Available: {model_names}")
                if "llama3.1:8b" not in model_names:
                    print("💡 Run: ollama pull llama3.1:8b")
                return False
        else:
            print(f"❌ Ollama server error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def validate_environment():
    """Validate all configurations with speed focus"""
    print("🚀 Validating FAST WhatsApp setup...")
    print(f"⚡ Fast Mode: {FAST_MODE}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"⏱️ Max Response Time: {MAX_RESPONSE_TIME}s")
    print(f"🔄 Rate Limit: {RATE_LIMIT_SECONDS}s")
    
    ollama_valid = validate_ollama_server()
    wa_valid = validate_whatsapp_config()
    
    if ollama_valid and wa_valid:
        print("🚀 ALL SYSTEMS READY FOR FAST WHATSAPP!")
        return True
    else:
        print("⚠️ Some configuration issues detected, but starting anyway...")
        return True  # Continue even with minor issues

# Validate on startup
validate_environment()
# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Redis Manager
class RedisManager:
    def __init__(self):
        self.redis_client = None
        self._connect()

    def _connect(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                password=REDIS_PASSWORD, decode_responses=False,
                socket_timeout=3, socket_connect_timeout=3, retry_on_timeout=True
            )
            self.redis_client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print(f"❌ Redis failed: {e}")
            self.redis_client = None

    def is_available(self) -> bool:
        return self.redis_client is not None

    def set_rate_limit(self, key: str, expire_seconds: int = 2):
        if not self.is_available():
            return False
        try:
            self.redis_client.setex(f"rate_limit:{key}", expire_seconds, "1")
            return True
        except Exception:
            return False

    def check_rate_limit(self, key: str) -> bool:
        if not self.is_available():
            return False
        try:
            return self.redis_client.exists(f"rate_limit:{key}") > 0
        except Exception:
            return False

redis_manager = RedisManager()

# Global knowledge base
KNOWLEDGE_BASE = {
    "documents": [],
    "vectorizer": None,
    "tfidf_matrix": None,
    "last_updated": None
}

# Initialize RAG chain
RAG_CHAIN = None

# Usage tracking
USAGE_STATS = {
    "messages_sent": 0,
    "messages_received": 0,
    "api_calls": 0,
    "start_time": time.time(),
    "daily_messages": 0,
    "last_reset": datetime.now().date(),
    "average_response_time": 0,
    "fast_responses": 0  # Responses under 5 seconds
}

# Track processed message IDs
PROCESSED_MESSAGE_IDS = set()

def track_usage(action: str, response_time: float = 0):
    """Enhanced usage tracking with speed metrics"""
    global USAGE_STATS
    
    today = datetime.now().date()
    if today != USAGE_STATS["last_reset"]:
        USAGE_STATS["daily_messages"] = 0
        USAGE_STATS["last_reset"] = today
    
    if action == "message_sent":
        USAGE_STATS["messages_sent"] += 1
        USAGE_STATS["daily_messages"] += 1
        
        # Track response times
        if response_time > 0:
            if response_time < 5.0:
                USAGE_STATS["fast_responses"] += 1
            
            # Update average
            total_responses = USAGE_STATS["messages_sent"]
            current_avg = USAGE_STATS["average_response_time"]
            USAGE_STATS["average_response_time"] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )
            
    elif action == "message_received":
        USAGE_STATS["messages_received"] += 1
    elif action == "api_call":
        USAGE_STATS["api_calls"] += 1

def rebuild_rag_chain():
    """Rebuild RAG chain with fast model"""
    global RAG_CHAIN
    try:
        if KNOWLEDGE_BASE["documents"] and RAG_SERVICES_AVAILABLE:
            print(f"🔄 Building FAST RAG chain with {MODEL_NAME}...")
            RAG_CHAIN = build_rag_chain_from_documents(
                KNOWLEDGE_BASE["documents"],
                ollama_model=MODEL_NAME,
                model_name=MODEL_NAME,
                enhanced_mode=True
            )
            KNOWLEDGE_BASE["last_updated"] = time.time()
            print(f"🚀 FAST RAG chain ready with {MODEL_NAME}")
        else:
            RAG_CHAIN = None
            print("⚠️ No documents or RAG services unavailable")
    except Exception as e:
        print(f"❌ Error building RAG chain: {e}")
        RAG_CHAIN = None

def ultra_fast_rag_response(query: str) -> str:
    """Ultra-fast response function with enhanced Devanagari language consistency"""
    start_time = time.time()
    print(f"⚡ FAST processing: '{query[:40]}...'")
    
    # Step 1: Check cache first for instant responses
    cached_response = fast_cache.get(query)
    if cached_response:
        response_time = time.time() - start_time
        print(f"💨 Instant cache response ({response_time:.2f}s)")
        return cached_response
    
    # Step 2: Enhanced Devanagari-aware language detection
    detected_language = detect_language(query)
    print(f"🗣️ Language detected: {detected_language}")
    
    # Step 3: Instant greeting responses with consistent Devanagari handling
    instant_greeting = get_instant_greeting(query, detected_language)
    if instant_greeting:
        fast_cache.set(query, instant_greeting)
        response_time = time.time() - start_time
        print(f"⚡ Instant greeting ({response_time:.2f}s)")
        return instant_greeting
    
    # Step 4: Enhanced validation checks with proper Devanagari error messages
    if not RAG_SERVICES_AVAILABLE:
        error_messages = {
            'hi': "एआई सेवाएं शुरू हो रही हैं। कृपया 30 सेकंड में कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "एआय सेवा सुरू होत आहे. कृपया 30 सेकंदांनी प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "AI services starting up. Please try again in 30 seconds.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        }
        return error_messages.get(detected_language, error_messages['en'])
    
    if not KNOWLEDGE_BASE["documents"]:
        error_messages = {
            'hi': "ज्ञान आधार उपलब्ध नहीं है। कृपया बाद में कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "ज्ञान आधार उपलब्ध नाही. कृपया नंतर प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "Knowledge base not available. Please try again later.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        }
        return error_messages.get(detected_language, error_messages['en'])
    
    # Step 5: Build/use RAG chain
    global RAG_CHAIN
    if RAG_CHAIN is None:
        try:
            print("🔄 Building FAST RAG chain...")
            rebuild_rag_chain()
        except Exception as e:
            print(f"❌ RAG chain build failed: {e}")
            error_messages = {
                'hi': "सिस्टम शुरू हो रहा है। कृपया 30 सेकंड में कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "सिस्टम सुरू होत आहे. कृपया 30 सेकंदांनी प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                'en': "System initializing. Please try again in 30 seconds.\n\n**For more details, please contact the 104/102 helpline numbers.**"
            }
            return error_messages.get(detected_language, error_messages['en'])
    
    # Step 6: Process with AI with enhanced language validation
    try:
        print(f"🔄 Processing with {MODEL_NAME}...")
        
        # Add language hint to query for better AI language matching
        language_hints = {
            'hi': f"[HINDI DEVANAGARI] {query}",
            'mr': f"[MARATHI DEVANAGARI] {query}", 
            'en': f"[ENGLISH] {query}"
        }
        enhanced_query = language_hints.get(detected_language, query)
        
        result = process_scheme_query_with_retry(RAG_CHAIN, enhanced_query, max_retries=1)
        result_text = result[0] if isinstance(result, tuple) else str(result)
        
        response_time = time.time() - start_time
        print(f"⚡ AI response in {response_time:.2f}s")
        
        if not result_text or len(result_text.strip()) < 10:
            error_messages = {
                'hi': "जानकारी नहीं मिली। कृपया दूसरे तरीके से पूछें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "माहिती मिळाली नाही. कृपया वेगळ्या पद्धतीने विचारा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                'en': "No information found. Please rephrase your question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
            }
            return error_messages.get(detected_language, error_messages['en'])
        
        # Step 7: Enhanced response validation with Devanagari script checking
        result_text = result_text.strip()
        
        # Convert markdown to WhatsApp formatting
        result_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', result_text)
        
        # Enhanced language validation for Devanagari scripts
        response_lang = detect_language(result_text[:200])  # Check first 200 chars
        
        # Check if response has appropriate script for detected language
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', result_text))
        
        language_mismatch = False
        if detected_language in ['hi', 'mr'] and not has_devanagari:
            language_mismatch = True
            print(f"⚠️ Script mismatch! {detected_language} query but no Devanagari in response")
        elif detected_language == 'en' and has_devanagari:
            language_mismatch = True
            print(f"⚠️ Script mismatch! English query but Devanagari in response")
        elif response_lang != detected_language:
            language_mismatch = True
            print(f"⚠️ Language mismatch! Query: {detected_language}, Response: {response_lang}")
        
        if language_mismatch:
            error_messages = {
                'hi': "भाषा की समस्या हुई। कृपया फिर से कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "भाषेची समस्या झाली. कृपया पुन्हा प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                'en': "Language processing issue. Please try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"
            }
            return error_messages.get(detected_language, error_messages['en'])
        
        # Ensure proper format with correct language
        if not validate_response_format(result_text, detected_language):
            result_text = handle_response_format(result_text, detected_language)
        
        # Cache for future speed
        fast_cache.set(query, result_text)
        
        total_time = time.time() - start_time
        print(f"🚀 Total response time: {total_time:.2f}s")
        
        return result_text
        
    except Exception as e:
        print(f"❌ AI processing failed: {e}")
        
        error_str = str(e).lower()
        if "timeout" in error_str:
            error_messages = {
                'hi': "जवाब धीमा है। छोटा प्रश्न पूछें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "उत्तर धीमे आहे. लहान प्रश्न विचारा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                'en': "Response slow. Ask shorter question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
            }
        else:
            error_messages = {
                'hi': "तकनीकी समस्या हुई। पुनः प्रयास करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "तांत्रिक समस्या झाली. पुन्हा प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                'en': "Technical issue occurred. Please try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"
            }
        
        return error_messages.get(detected_language, error_messages['en'])

# FastAPI setup
class QueryRequest(BaseModel):
    input_text: str
    model: str = "llama3.1:8b"
    enhanced_mode: bool = True
    voice_lang_pref: str = "auto"
    session_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting FAST WhatsApp AI application...")
    print(f"⚡ Model: {MODEL_NAME}")
    print(f"🔄 Rate Limit: {RATE_LIMIT_SECONDS}s")
    yield
    if redis_manager.is_available():
        redis_manager.redis_client.close()
    print("🛑 Application shutting down...")

app = FastAPI(title="FAST WhatsApp AI - Production Ready", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    try:
        cache_stats = fast_cache.get_stats()
        return {
            "message": "FAST WhatsApp AI - Production Ready",
            "model": MODEL_NAME,
            "fast_mode": FAST_MODE,
            "max_response_time": MAX_RESPONSE_TIME,
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "cache_stats": cache_stats,
            "knowledge_base_documents": len(KNOWLEDGE_BASE["documents"]),
            "rag_chain_ready": RAG_CHAIN is not None,
            "services_available": {
                "rag_services": RAG_SERVICES_AVAILABLE,
                "transcription": TRANSCRIPTION_AVAILABLE,
                "redis": redis_manager.is_available()
            },
            "speed_optimization": "ENABLED",
            "enhanced_devanagari_detection": "ENABLED",
            "docs": "/docs",
            "health": "/health/"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health/")
async def health_check():
    """Fast health check"""
    cache_stats = fast_cache.get_stats()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "fast_mode": FAST_MODE,
        "cache_stats": cache_stats,
        "knowledge_base_documents": len(KNOWLEDGE_BASE["documents"]),
        "redis_available": redis_manager.is_available(),
        "rag_chain_ready": RAG_CHAIN is not None,
        "enhanced_language_detection": True,
        "timestamp": time.time(),
        "speed_optimized": True
    }

@app.post("/upload/")
async def upload_knowledge_files(
    pdf_file: Optional[UploadFile] = File(None),
    txt_file: Optional[UploadFile] = File(None)
):
    """Fast file upload with optimized processing"""
    try:
        if not pdf_file and not txt_file:
            return JSONResponse(
                status_code=400, 
                content={"error": "Please upload at least one PDF or TXT file"}
            )

        saved_files = []
        start_time = time.time()
        
        # Process PDF file
        if pdf_file:
            pdf_bytes = await pdf_file.read()
            os.makedirs("uploads", exist_ok=True)
            pdf_filename = os.path.join("uploads", f"{int(time.time())}_{pdf_file.filename}")
            with open(pdf_filename, "wb") as f:
                f.write(pdf_bytes)
                saved_files.append(pdf_filename)
            
            # Fast PDF reading
            try:
                with open(pdf_filename, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    KNOWLEDGE_BASE["documents"].append({"filename": pdf_filename, "content": text})
            except Exception as e:
                print(f"❌ PDF reading error: {e}")

        # Process TXT file
        if txt_file:
            txt_bytes = await txt_file.read()
            os.makedirs("uploads", exist_ok=True)
            txt_filename = os.path.join("uploads", f"{int(time.time())}_{txt_file.filename}")
            with open(txt_filename, "wb") as f:
                f.write(txt_bytes)
                saved_files.append(txt_filename)
            
            # Fast TXT reading
            try:
                with open(txt_filename, "r", encoding="utf-8") as f:
                    text = f.read()
                    KNOWLEDGE_BASE["documents"].append({"filename": txt_filename, "content": text})
            except Exception as e:
                print(f"❌ TXT reading error: {e}")

        # Fast RAG chain rebuild
        rebuild_rag_chain()
        
        processing_time = time.time() - start_time
        print(f"⚡ Files processed in {processing_time:.2f}s")

        return {
            "message": "Files uploaded and knowledge base updated",
            "saved_files": saved_files,
            "total_documents": len(KNOWLEDGE_BASE["documents"]),
            "processing_time_seconds": round(processing_time, 2),
            "model": MODEL_NAME,
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query/")
async def query_knowledge_base(req: QueryRequest):
    """Fast query processing with enhanced language detection"""
    try:
        query = req.input_text.strip()
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query cannot be empty"}
            )

        if not KNOWLEDGE_BASE["documents"]:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No knowledge base available. Upload files first.",
                    "upload_endpoint": "/upload/"
                }
            )

        start_time = time.time()
        
        # Fast response generation with enhanced language detection
        response = ultra_fast_rag_response(query)
        
        response_time = time.time() - start_time
        user_language = detect_language(query)
        
        # Validate response format quickly
        if not validate_response_format(response, user_language):
            response = handle_response_format(response, user_language)
        
        return {
            "query": query,
            "response": response,
            "response_time_seconds": round(response_time, 2),
            "language_detected": user_language,
            "has_devanagari": bool(re.search(r'[\u0900-\u097F]', query)),
            "knowledge_base_size": len(KNOWLEDGE_BASE["documents"]),
            "model": MODEL_NAME,
            "fast_mode": FAST_MODE,
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# WhatsApp webhook endpoints
@app.get("/webhook")
async def verify_whatsapp(request: Request):
    """WhatsApp verification endpoint"""
    try:
        params = dict(request.query_params)
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")
        
        print(f"[VERIFY] mode={mode}, token={'***' if token else 'None'}")
        
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            print("[OK] VERIFICATION SUCCESS!")
            return int(challenge)
        else:
            print("[ERROR] VERIFICATION FAILED!")
            return JSONResponse(status_code=403, content={"error": "Verification failed"})
            
    except Exception as e:
        print(f"[CRASH] Verification error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """ULTRA-FAST WhatsApp message handler with enhanced language processing"""
    global PROCESSED_MESSAGE_IDS
    
    try:
        data = await request.json()
        
        # Fast message extraction
        if not (data.get("entry") and data["entry"][0].get("changes")):
            return {"status": "ok"}
        
        change = data["entry"][0]["changes"][0]
        value = change.get("value", {})
        
        # Skip status updates quickly
        if "statuses" in value:
            return {"status": "ok"}
        
        # Fast message processing
        messages = value.get("messages")
        if not messages:
            return {"status": "ok"}
        
        message_obj = messages[0]
        message_type = message_obj.get("type")
        user_number = message_obj.get("from")
        message_id = message_obj.get("id")
        
        print(f"📝 MSG: {message_type} from {user_number}")
        
        # Quick duplicate check
        if message_id in PROCESSED_MESSAGE_IDS:
            print(f"🔄 DUPLICATE: {message_id}")
            return {"status": "duplicate"}
        
        # Skip our own messages
        if user_number == WHATSAPP_PHONE_NUMBER_ID:
            return {"status": "ignored_own"}
        
        # Process text messages only
        if message_type == "text":
            user_msg = message_obj.get("text", {}).get("body", "").strip()
            
            if user_msg:
                print(f"💬 Processing: '{user_msg[:30]}...'")
                
                # Fast tracking
                track_usage("message_received")
                PROCESSED_MESSAGE_IDS.add(message_id)
                
                # Keep only recent 500 message IDs for memory efficiency
                if len(PROCESSED_MESSAGE_IDS) > 500:
                    old_ids = list(PROCESSED_MESSAGE_IDS)[-250:]
                    PROCESSED_MESSAGE_IDS = set(old_ids)
                
                # Fast rate limiting
                rate_limit_key = f"whatsapp_{user_number}"
                if redis_manager.check_rate_limit(rate_limit_key):
                    print(f"⏱️ Rate limited: {user_number}")
                    return {"status": "rate_limited"}
                
                # Set fast rate limit
                redis_manager.set_rate_limit(rate_limit_key, RATE_LIMIT_SECONDS)
                
                # ULTRA-FAST response generation with enhanced language detection
                start_time = time.time()
                detected_lang = detect_language(user_msg)
                response = ultra_fast_rag_response(user_msg)
                response_time = time.time() - start_time
                
                print(f"⚡ Response in {response_time:.2f}s (Language: {detected_lang})")
                
                # Validate response quickly - NO TRUNCATION
                if not response or len(response.strip()) < 5:
                    fallback_messages = {
                        'hi': "कृपया अपना प्रश्न दोबारा पूछें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                        'mr': "कृपया तुमचा प्रश्न पुन्हा विचारा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
                        'en': "Please try rephrasing your question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
                    }
                    response = fallback_messages.get(detected_lang, fallback_messages['en'])
                
                # Enhanced WhatsApp send with retries
                success = send_whatsapp_message(user_number, response)
                
                if success:
                    track_usage("message_sent", response_time)
                    print(f"📤 SUCCESS in {response_time:.2f}s")
                else:
                    print(f"❌ SEND FAILED after all retries")
                
                return {
                    "status": "processed",
                    "message_id": message_id,
                    "detected_language": detected_lang,
                    "has_devanagari": bool(re.search(r'[\u0900-\u097F]', user_msg)),
                    "response_time_seconds": round(response_time, 2),
                    "send_success": success,
                    "model": MODEL_NAME,
                    "fast_mode": FAST_MODE
                }
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"💥 Webhook error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/usage/stats")
async def get_usage_stats():
    """Enhanced usage statistics with speed and language metrics"""
    uptime_hours = (time.time() - USAGE_STATS["start_time"]) / 3600
    cache_stats = fast_cache.get_stats()
    
    fast_response_rate = 0
    if USAGE_STATS["messages_sent"] > 0:
        fast_response_rate = (USAGE_STATS["fast_responses"] / USAGE_STATS["messages_sent"]) * 100
    
    return {
        "current_usage": {
            "messages_sent_total": USAGE_STATS["messages_sent"],
            "messages_received_total": USAGE_STATS["messages_received"],
            "daily_messages": USAGE_STATS["daily_messages"],
            "api_calls": USAGE_STATS["api_calls"]
        },
        "speed_metrics": {
            "average_response_time_seconds": round(USAGE_STATS["average_response_time"], 2),
            "fast_responses_under_5s": USAGE_STATS["fast_responses"],
            "fast_response_rate_percent": round(fast_response_rate, 1),
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["cache_size"]
        },
        "language_capabilities": {
            "supported_languages": ["English", "Hindi", "Marathi"],
            "language_codes": ["en", "hi", "mr"],
            "enhanced_devanagari_detection": True,
            "romanized_text_support": True
        },
        "system_info": {
            "model": MODEL_NAME,
            "fast_mode": FAST_MODE,
            "max_response_time": MAX_RESPONSE_TIME,
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "uptime_hours": round(uptime_hours, 2)
        },
        "performance_status": {
            "excellent": fast_response_rate > 80,
            "good": 60 <= fast_response_rate <= 80,
            "needs_optimization": fast_response_rate < 60
        }
    }

@app.get("/speed/test")
async def speed_test():
    """Test response speed with enhanced language detection"""
    try:
        test_queries = [
            "Hello",
            "What is PM Awas Yojana?",
            "नमस्ते",
            "सरकारी योजना क्या है?",
            "मला माहिती पाहिजे",
            "सरकारी योजना बद्दल सांगा",
            "Jssk baddal mahiti dya"
        ]
        
        results = []
        
        for query in test_queries:
            start_time = time.time()
            response = ultra_fast_rag_response(query)
            response_time = time.time() - start_time
            detected_lang = detect_language(query)
            
            results.append({
                "query": query,
                "detected_language": detected_lang,
                "response_time_seconds": round(response_time, 2),
                "response_length": len(response),
                "has_devanagari_input": bool(re.search(r'[\u0900-\u097F]', query)),
                "has_devanagari_output": bool(re.search(r'[\u0900-\u097F]', response)),
                "script_consistency": (detected_lang == 'en' and not bool(re.search(r'[\u0900-\u097F]', response))) or 
                                    (detected_lang in ['hi', 'mr'] and bool(re.search(r'[\u0900-\u097F]', response))),
                "status": "fast" if response_time < 5 else "slow"
            })
        
        avg_time = sum(r["response_time_seconds"] for r in results) / len(results)
        script_accuracy = sum(1 for r in results if r["script_consistency"]) / len(results) * 100
        
        return {
            "test_results": results,
            "summary": {
                "average_response_time": round(avg_time, 2),
                "script_consistency_rate": round(script_accuracy, 1),
                "total_tests": len(results),
                "fast_responses": sum(1 for r in results if r["status"] == "fast")
            },
            "model": MODEL_NAME,
            "fast_mode": FAST_MODE,
            "performance_rating": "excellent" if avg_time < 3 else "good" if avg_time < 8 else "needs_optimization",
            "language_detection_accuracy": "enhanced_devanagari_support",
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/cache/clear")
async def clear_all_caches():
    """Clear all caches for fresh start"""
    try:
        # Clear fast cache
        old_stats = fast_cache.get_stats()
        fast_cache.cache.clear()
        fast_cache.stats = {"hits": 0, "misses": 0}
        
        # Clear RAG cache if available
        if RAG_SERVICES_AVAILABLE:
            try:
                clear_query_cache()
            except:
                pass
        
        return {
            "message": "All caches cleared successfully",
            "previous_cache_stats": old_stats,
            "timestamp": time.time(),
            "cache_types_cleared": ["fast_response_cache", "rag_query_cache"]
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/performance")
async def debug_performance():
    """Performance debugging information with language detection metrics"""
    try:
        # Test Ollama speed
        start_time = time.time()
        test_payload = {
            "model": MODEL_NAME,
            "prompt": "Test speed",
            "stream": False,
            "options": {"num_predict": 5}
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=test_payload,
            timeout=10
        )
        
        ollama_response_time = time.time() - start_time
        
        cache_stats = fast_cache.get_stats()
        
        # Test language detection accuracy
        test_texts = {
            "english_simple": "What is the PM Awas Yojana scheme?",
            "english_complex": "Tell me about government housing schemes and eligibility criteria",
            "hindi_simple": "मुझे सरकारी योजना के बारे में जानकारी चाहिए।",
            "hindi_complex": "प्रधानमंत्री आवास योजना की पात्रता क्या है?",
            "marathi_simple": "मला सरकारी योजनेची माहिती पाहिजे.",
            "marathi_complex": "मुख्यमंत्री योजनेची अर्ज करण्याची प्रक्रिया काय आहे?",
            "mixed_english": "What is सरकारी योजना benefits?",
            "mixed_marathi": "मला government scheme माहिती हवी"
        }
        
        language_detection_results = {}
        for test_name, text in test_texts.items():
            detected = detect_language(text)
            expected = test_name.split('_')[0]
            expected_code = {'english': 'en', 'hindi': 'hi', 'marathi': 'mr', 'mixed': 'en'}
            
            language_detection_results[test_name] = {
                "text": text,
                "expected_category": expected,
                "detected_code": detected,
                "has_devanagari": bool(re.search(r'[\u0900-\u097F]', text)),
                "correct": detected == expected_code.get(expected, expected)
            }
        
        accuracy = sum(1 for r in language_detection_results.values() if r["correct"]) / len(language_detection_results)
        
        return {
            "performance_metrics": {
                "ollama_response_time_seconds": round(ollama_response_time, 2),
                "ollama_status": "online" if response.status_code == 200 else "error",
                "cache_performance": cache_stats,
                "model": MODEL_NAME,
                "fast_mode": FAST_MODE
            },
            "language_detection": {
                "test_results": language_detection_results,
                "accuracy_percentage": round(accuracy * 100, 1),
                "enhanced_devanagari": True,
                "supported_scripts": ["Roman", "Devanagari"],
                "supported_languages": ["English", "Hindi", "Marathi"]
            },
            "system_status": {
                "knowledge_base_documents": len(KNOWLEDGE_BASE["documents"]),
                "rag_chain_ready": RAG_CHAIN is not None,
                "redis_available": redis_manager.is_available(),
                "rag_services_available": RAG_SERVICES_AVAILABLE
            },
            "speed_recommendations": {
                "current_model_speed": "excellent" if "8b" in MODEL_NAME else "slow",
                "recommended_for_whatsapp": "llama3.1:8b",
                "current_rate_limit": f"{RATE_LIMIT_SECONDS}s",
                "recommended_rate_limit": "1-2s for best UX"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/whatsapp/test")
async def test_whatsapp_message(request: Request):
    """Test WhatsApp sending functionality"""
    try:
        data = await request.json()
        phone_number = data.get("phone_number")
        message = data.get("message", "Test message from FAST WhatsApp AI with Enhanced Devanagari Support")
        
        if not phone_number:
            return JSONResponse(status_code=400, content={"error": "phone_number is required"})
        
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number.lstrip('+')
        
        start_time = time.time()
        success = send_whatsapp_message(phone_number, message)
        send_time = time.time() - start_time
        
        if success:
            track_usage("message_sent", send_time)
        
        return {
            "success": success,
            "phone_number": phone_number,
            "message": message,
            "send_time_seconds": round(send_time, 2),
            "timestamp": time.time(),
            "model": MODEL_NAME,
            "enhanced_features": ["devanagari_detection", "language_consistency", "fast_response"]
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/language/test")
async def test_language_detection():
    """Test enhanced language detection capabilities"""
    try:
        test_cases = [
            # English tests
            {"text": "What is PM Awas Yojana?", "expected": "en", "type": "english_simple"},
            {"text": "Tell me about government schemes and eligibility", "expected": "en", "type": "english_complex"},
            
            # Hindi tests
            {"text": "मुझे जानकारी चाहिए", "expected": "hi", "type": "hindi_simple"},
            {"text": "सरकारी योजना के बारे में बताएं", "expected": "hi", "type": "hindi_complex"},
            {"text": "प्रधानमंत्री आवास योजना क्या है?", "expected": "hi", "type": "hindi_scheme"},
            {"text": "मैं आवेदन कैसे करूं?", "expected": "hi", "type": "hindi_application"},
            
            # Marathi tests
            {"text": "मला माहिती पाहिजे", "expected": "mr", "type": "marathi_simple"},
            {"text": "सरकारी योजना बद्दल सांगा", "expected": "mr", "type": "marathi_complex"},
            {"text": "मुख्यमंत्री योजना कशी आहे?", "expected": "mr", "type": "marathi_scheme"},
            {"text": "मी अर्ज कसा करावा?", "expected": "mr", "type": "marathi_application"},
            
            # Romanized tests
            {"text": "kya hai PM Awas Yojana", "expected": "hi", "type": "hindi_romanized"},
            {"text": "mahiti pahije government scheme", "expected": "mr", "type": "marathi_romanized"},
            {"text": "mujhe chahiye information", "expected": "hi", "type": "hindi_mixed_roman"},
            
            # Mixed script tests
            {"text": "What is सरकारी योजना?", "expected": "en", "type": "mixed_english_dominant"},
            {"text": "मला government scheme माहिती हवी", "expected": "mr", "type": "mixed_marathi_dominant"},
            {"text": "Tell me about योजना details", "expected": "en", "type": "mixed_english_query"}
        ]
        
        results = []
        correct_detections = 0
        
        for test_case in test_cases:
            detected = detect_language(test_case["text"])
            is_correct = detected == test_case["expected"]
            if is_correct:
                correct_detections += 1
            
            results.append({
                "text": test_case["text"],
                "expected": test_case["expected"],
                "detected": detected,
                "correct": is_correct,
                "test_type": test_case["type"],
                "has_devanagari": bool(re.search(r'[\u0900-\u097F]', test_case["text"])),
                "has_english": bool(re.search(r'[a-zA-Z]', test_case["text"])),
                "script_type": "mixed" if (bool(re.search(r'[\u0900-\u097F]', test_case["text"])) and 
                                         bool(re.search(r'[a-zA-Z]', test_case["text"]))) else
                             "devanagari" if bool(re.search(r'[\u0900-\u097F]', test_case["text"])) else "roman"
            })
        
        accuracy = (correct_detections / len(test_cases)) * 100
        
        # Group results by language
        language_breakdown = {}
        for result in results:
            expected_lang = result["expected"]
            if expected_lang not in language_breakdown:
                language_breakdown[expected_lang] = {"total": 0, "correct": 0}
            language_breakdown[expected_lang]["total"] += 1
            if result["correct"]:
                language_breakdown[expected_lang]["correct"] += 1
        
        # Calculate per-language accuracy
        for lang in language_breakdown:
            total = language_breakdown[lang]["total"]
            correct = language_breakdown[lang]["correct"]
            language_breakdown[lang]["accuracy"] = round((correct / total) * 100, 1) if total > 0 else 0
        
        return {
            "test_summary": {
                "total_tests": len(test_cases),
                "correct_detections": correct_detections,
                "overall_accuracy_percentage": round(accuracy, 1),
                "enhanced_devanagari_support": True,
                "language_breakdown": language_breakdown
            },
            "detailed_results": results,
            "language_capabilities": {
                "supported_languages": ["English", "Hindi", "Marathi"],
                "language_codes": ["en", "hi", "mr"],
                "script_support": ["Roman", "Devanagari"],
                "mixed_script_handling": True,
                "romanized_text_detection": True,
                "weighted_scoring_system": True
            },
            "detection_features": {
                "strong_indicators": "Weighted 2x (verbs, pronouns, particles)",
                "weak_indicators": "Weighted 1x (common words)",
                "character_patterns": "Marathi conjuncts vs Hindi patterns",
                "fallback_mechanisms": "Multiple levels of detection"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/system/info")
async def get_system_info():
    """Get comprehensive system information"""
    try:
        cache_stats = fast_cache.get_stats()
        uptime_hours = (time.time() - USAGE_STATS["start_time"]) / 3600
        
        return {
            "application_info": {
                "name": "FAST WhatsApp AI - Production Ready",
                "version": "2.1.0",
                "enhanced_features": [
                    "Advanced Devanagari Language Detection",
                    "Ultra-Fast Response Processing",
                    "Enhanced Caching System",
                    "Production-Ready Error Handling",
                    "Script Consistency Validation",
                    "Weighted Language Scoring"
                ]
            },
            "language_capabilities": {
                "supported_languages": ["English", "Hindi", "Marathi"],
                "language_codes": ["en", "hi", "mr"],
                "script_support": ["Roman", "Devanagari"],
                "detection_method": "Enhanced weighted scoring with character patterns",
                "advanced_detection": True,
                "romanized_support": True,
                "mixed_script_handling": True,
                "script_consistency_validation": True
            },
            "performance_metrics": {
                "model": MODEL_NAME,
                "fast_mode": FAST_MODE,
                "average_response_time": USAGE_STATS["average_response_time"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "uptime_hours": round(uptime_hours, 2),
                "total_messages_processed": USAGE_STATS["messages_sent"] + USAGE_STATS["messages_received"]
            },
            "system_status": {
                "rag_chain_ready": RAG_CHAIN is not None,
                "knowledge_base_size": len(KNOWLEDGE_BASE["documents"]),
                "redis_available": redis_manager.is_available(),
                "services_available": {
                    "rag_services": RAG_SERVICES_AVAILABLE,
                    "transcription": TRANSCRIPTION_AVAILABLE,
                    "enhanced_language_detection": True
                }
            },
            "configuration": {
                "max_response_time": MAX_RESPONSE_TIME,
                "rate_limit_seconds": RATE_LIMIT_SECONDS,
                "cache_ttl": CACHE_TTL,
                "cache_size": cache_stats["cache_size"],
                "fast_mode_enabled": FAST_MODE
            },
            "api_endpoints": {
                "core_endpoints": {
                    "health_check": "/health/",
                    "upload_files": "/upload/",
                    "query_knowledge": "/query/",
                    "whatsapp_webhook": "/webhook"
                },
                "testing_endpoints": {
                    "speed_test": "/speed/test",
                    "language_test": "/language/test",
                    "whatsapp_test": "/whatsapp/test"
                },
                "admin_endpoints": {
                    "system_stats": "/usage/stats",
                    "debug_performance": "/debug/performance",
                    "clear_cache": "/cache/clear",
                    "system_info": "/system/info"
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/admin/reset")
async def admin_reset_system():
    """Admin endpoint to reset system state"""
    try:
        global RAG_CHAIN, PROCESSED_MESSAGE_IDS
        
        # Clear all caches
        fast_cache.cache.clear()
        fast_cache.stats = {"hits": 0, "misses": 0}
        
        if RAG_SERVICES_AVAILABLE:
            try:
                clear_query_cache()
            except:
                pass
        
        # Reset RAG chain
        RAG_CHAIN = None
        
        # Clear processed message IDs
        PROCESSED_MESSAGE_IDS.clear()
        
        # Reset usage stats
        global USAGE_STATS
        USAGE_STATS = {
            "messages_sent": 0,
            "messages_received": 0,
            "api_calls": 0,
            "start_time": time.time(),
            "daily_messages": 0,
            "last_reset": datetime.now().date(),
            "average_response_time": 0,
            "fast_responses": 0
        }
        
        return {
            "message": "System reset completed successfully",
            "reset_components": [
                "response_cache",
                "rag_query_cache", 
                "rag_chain",
                "processed_message_ids",
                "usage_statistics"
            ],
            "enhanced_features_maintained": [
                "devanagari_language_detection",
                "script_consistency_validation",
                "weighted_scoring_system"
            ],
            "timestamp": time.time(),
            "system_ready": True
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/admin/logs")
async def get_recent_logs():
    """Get recent system logs"""
    try:
        log_file = "whatsapp.log"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Get last 50 lines
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                
            return {
                "recent_logs": [line.strip() for line in recent_lines],
                "total_lines": len(lines),
                "showing_last": len(recent_lines),
                "log_file": log_file,
                "enhanced_logging": True,
                "timestamp": time.time()
            }
        else:
            return {
                "message": "No log file found",
                "log_file": log_file,
                "timestamp": time.time()
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ULTRA-FAST WhatsApp AI Server...")
    print("📊 Configuration Summary:")
    print(f"   ⚡ Model: {MODEL_NAME}")
    print(f"   🔥 Fast Mode: {FAST_MODE}")
    print(f"   ⏱️ Max Response Time: {MAX_RESPONSE_TIME}s")
    print(f"   🔄 Rate Limit: {RATE_LIMIT_SECONDS}s")
    print(f"   💾 Cache TTL: {CACHE_TTL}s")

    uvicorn.run(app, host="0.0.0.0", port=8080)