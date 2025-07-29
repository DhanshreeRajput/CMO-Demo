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

# SPEED OPTIMIZATION SETTINGS
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"
MAX_RESPONSE_TIME = int(os.getenv("MAX_RESPONSE_TIME", 45))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))
CACHE_TTL = int(os.getenv("CACHE_TTL", 1800))

# IMPROVED LANGUAGE DETECTION
def detect_language(text):
    """Enhanced language detection for Marathi, Hindi, and English"""
    try:
        clean_text = text.strip().lower()
        
        # Check for Devanagari characters
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', clean_text))
        english_chars = bool(re.search(r'[a-zA-Z]', clean_text))

        if hindi_chars and not english_chars:
            # Enhanced Marathi vs Hindi detection
            marathi_words = ['baddal', 'mahiti', 'dya', 'kasa', 'kara', 'आहे', 'तुम्ही', 'मी', 'माहिती', 'येथे', 'करा', 'कसा', 'बद्दल']
            hindi_words = ['ke', 'liye', 'kaise', 'karna', 'है', 'आपके', 'जानकारी', 'कैसे', 'करना', 'के लिए']
            
            marathi_count = sum(1 for word in marathi_words if word in clean_text)
            hindi_count = sum(1 for word in hindi_words if word in clean_text)
            
            if marathi_count > hindi_count:
                return 'marathi'
            return 'hindi'
        elif english_chars:
            return 'english'
        else:
            return 'english'  # Default
            
    except Exception as e:
        logging.error(f"Language detection error: {e}")
        return 'english'

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
    """Fast response formatting for WhatsApp"""
    helpline_endings = {
        'english': "\n\n**For more details, please contact the 104/102 helpline numbers.**",
        'hindi': "\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
        'marathi': "\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
    }

    # Quick formatting check
    has_ending = any(ending.strip() in response_text for ending in helpline_endings.values())
    
    if not has_ending:
        ending = helpline_endings.get(user_language.lower(), helpline_endings['english'])
        response_text += ending

    return response_text

def validate_response_format(response_text, user_language):
    """Quick response validation"""
    return len(response_text) > 20 and ('104' in response_text or '102' in response_text)

# FAST GREETING RESPONSES (bypass AI completely)
def get_instant_greeting(query: str, language: str) -> Optional[str]:
    """Get instant greeting responses without AI"""
    query_lower = query.strip().lower()
    
    greetings = ['hi', 'hello', 'hey', 'namaste', 'नमस्ते', 'नमस्कार', 'good morning', 'good afternoon']
    
    if query_lower in greetings or any(greet in query_lower for greet in greetings):
        if language == 'hindi':
            return "नमस्ते! मैं आपकी सरकारी योजनाओं में कैसे मदद कर सकती हूँ? 😊\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**"
        elif language == 'marathi':
            return "नमस्कार! मी तुम्हाला सरकारी योजनांमध्ये कशी मदत करू शकते? 😊\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
        else:
            return "Hello! How can I help you with government schemes today? 😊\n\n**For more details, please contact the 104/102 helpline numbers.**"
    
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
    """Ultra-fast response function optimized for WhatsApp"""
    start_time = time.time()
    print(f"⚡ FAST processing: '{query[:40]}...'")
    
    # Step 1: Check cache first for instant responses
    cached_response = fast_cache.get(query)
    if cached_response:
        response_time = time.time() - start_time
        print(f"💨 Instant cache response ({response_time:.2f}s)")
        return cached_response
    
    # Step 2: Enhanced language detection
    language = detect_language(query)
    print(f"🗣️ Language: {language}")
    
    # Step 3: Instant greeting responses (bypass AI)
    instant_greeting = get_instant_greeting(query, language)
    if instant_greeting:
        fast_cache.set(query, instant_greeting)
        response_time = time.time() - start_time
        print(f"⚡ Instant greeting ({response_time:.2f}s)")
        return instant_greeting
    
    # Step 4: Quick validation checks
    if not RAG_SERVICES_AVAILABLE:
        error_response = "AI services starting up. Please try again in 30 seconds."
        return handle_response_format(error_response, language)
    
    if not KNOWLEDGE_BASE["documents"]:
        error_messages = {
            'hindi': "ज्ञान आधार उपलब्ध नहीं। कृपया बाद में कोशिश करें।",
            'marathi': "ज्ञान आधार उपलब्ध नाही. कृपया नंतर प्रयत्न करा.",
            'english': "Knowledge base not available. Please try again later."
        }
        error_response = error_messages.get(language, error_messages['english'])
        return handle_response_format(error_response, language)
    
    # Step 5: Build/use RAG chain
    global RAG_CHAIN
    if RAG_CHAIN is None:
        try:
            print("🔄 Building FAST RAG chain...")
            rebuild_rag_chain()
        except Exception as e:
            print(f"❌ RAG chain build failed: {e}")
            error_response = "System initializing. Please try again in 30 seconds."
            return handle_response_format(error_response, language)
    
    # Step 6: Process with AI (with timeout)
    try:
        print(f"🔄 Processing with {MODEL_NAME}...")
        
        result = process_scheme_query_with_retry(RAG_CHAIN, query, max_retries=1)
        result_text = result[0] if isinstance(result, tuple) else str(result)
        
        response_time = time.time() - start_time
        print(f"⚡ AI response in {response_time:.2f}s")
        
        if not result_text or len(result_text.strip()) < 10:
            error_messages = {
                'hindi': "जानकारी नहीं मिली। कृपया दूसरे तरीके से पूछें।",
                'marathi': "माहिती मिळाली नाही. कृपया वेगळ्या पद्धतीने विचारा.",
                'english': "No information found. Please rephrase your question."
            }
            error_response = error_messages.get(language, error_messages['english'])
            return handle_response_format(error_response, language)
        
        # Step 7: Format and cache response
        result_text = result_text.strip()
        
        # Convert markdown to WhatsApp formatting
        result_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', result_text)
        
        # Ensure proper format
        if not validate_response_format(result_text, language):
            result_text = handle_response_format(result_text, language)
        
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
                'hindi': "Response धीमा है। छोटा प्रश्न पूछें।",
                'marathi': "प्रतिसाद धीमा आहे. लहान प्रश्न विचारा.",
                'english': "Response slow. Ask shorter question."
            }
        else:
            error_messages = {
                'hindi': "तकनीकी समस्या। पुनः प्रयास करें।",
                'marathi': "तांत्रिक समस्या. पुन्हा प्रयत्न करा.",
                'english': "Technical issue. Please try again."
            }
        
        error_response = error_messages.get(language, error_messages['english'])
        return handle_response_format(error_response, language)

# COMPLETE FASTAPI ENDPOINTS - PART 2
# Add this to the end of the previous file

# FastAPI setup
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
    """Fast query processing"""
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
        
        # Fast response generation
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
    """ULTRA-FAST WhatsApp message handler with enhanced error handling"""
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
                
                # ULTRA-FAST response generation
                start_time = time.time()
                response = ultra_fast_rag_response(user_msg)
                response_time = time.time() - start_time
                
                print(f"⚡ Response in {response_time:.2f}s")
                
                # Validate response quickly - NO TRUNCATION
                if not response or len(response.strip()) < 5:
                    response = "Please try rephrasing your question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
                
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
    """Enhanced usage statistics with speed metrics"""
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
    """Test response speed"""
    try:
        test_queries = [
            "Hello",
            "What is PM Awas Yojana?",
            "नमस्ते",
            "सरकारी योजना क्या है?",
            "Jssk baddal mahiti dya"
        ]
        
        results = []
        
        for query in test_queries:
            start_time = time.time()
            response = ultra_fast_rag_response(query)
            response_time = time.time() - start_time
            
            results.append({
                "query": query,
                "detected_language": detect_language(query),
                "response_time_seconds": round(response_time, 2),
                "response_length": len(response),
                "status": "fast" if response_time < 5 else "slow"
            })
        
        avg_time = sum(r["response_time_seconds"] for r in results) / len(results)
        
        return {
            "test_results": results,
            "average_response_time": round(avg_time, 2),
            "model": MODEL_NAME,
            "fast_mode": FAST_MODE,
            "performance_rating": "excellent" if avg_time < 3 else "good" if avg_time < 8 else "needs_optimization",
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
    """Performance debugging information"""
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
        
        return {
            "performance_metrics": {
                "ollama_response_time_seconds": round(ollama_response_time, 2),
                "ollama_status": "online" if response.status_code == 200 else "error",
                "cache_performance": cache_stats,
                "model": MODEL_NAME,
                "fast_mode": FAST_MODE
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
        message = data.get("message", "Test message from FAST WhatsApp AI")
        
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
            "model": MODEL_NAME
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
    print("")
    print("🎯 Performance Targets:")
    print("   • Greetings: <0.5s (cached)")
    print("   • Simple queries: 2-5s")
    print("   • Complex queries: 5-15s")
    print("   • Cache hit rate: >80%")
    print("")
    print("🔧 Quick Setup Commands:")
    print("   1. ollama pull llama3.1:8b")
    print("   2. Upload knowledge documents via /upload/")
    print("   3. Test with: curl localhost:8080/speed/test")
    print("")
    print("🌐 Network Resilience:")
    print("   • Auto-retry on connection failures")
    print("   • Exponential backoff for rate limits")
    print("   • Extended timeouts for stability")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8080)