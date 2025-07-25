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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import aiohttp
import asyncio
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whatsapp.log'),
        logging.StreamHandler()
    ]
)

# Add these validation functions here
def detect_language(text):
    """Detect language based on character scripts or fallback logic."""
    try:
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', text))
        english_chars = bool(re.search(r'[a-zA-Z]', text))

        if hindi_chars and not english_chars:
            marathi_keywords = ['आहे', 'करा', 'होणार', 'येथे', 'तुमच्या']
            if any(keyword in text for keyword in marathi_keywords):
                return 'marathi'
            return 'hindi'
        elif english_chars:
            return 'english'
        else:
            # Fallback logic
            if 'देवनागरी' in text:
                return 'hindi'
            elif 'मराठी' in text:
                return 'marathi'
            else:
                return 'english'
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        return 'unknown'

def handle_response_format(response_text, user_language):
    """Validate and fix response format if necessary."""
    helpline_endings = {
        'english': "\n\n**For more details, please contact the 104/102 helpline numbers.**",
        'hindi': "\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
        'marathi': "\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
    }

    required_elements = {
        'english': ['**For more details, please contact the 104/102 helpline numbers.**'],
        'hindi': ['**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**'],
        'marathi': ['**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**']
    }

    has_ending = any(ending in response_text for ending in required_elements.get(user_language.lower(), []))
    has_formatting = '**' in response_text and ('##' in response_text or '#' in response_text)
    word_count = len(response_text.split())
    min_length = word_count >= 50

    if not (has_ending and has_formatting and min_length):
        ending = helpline_endings.get(user_language.lower(), helpline_endings['english'])
        if ending.strip() not in response_text:
            response_text += ending

    logging.info(f"Response format handled for language: {user_language}")
    return response_text

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Environment variables - Updated for Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
AI_PROVIDER = os.getenv("AI_PROVIDER", "google")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Import services with fallback
try:
    from core.rag_services import (
        detect_language as transcription_detect_language,
        build_rag_chain_from_documents,
        build_rag_chain_with_model_choice,
        process_scheme_query_with_retry,
        clear_query_cache,
        get_model_options
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
    print("⚠️ transcription.py not found, will create fallback")
    TRANSCRIPTION_AVAILABLE = False

def validate_environment():
    """Validate all required environment variables"""
    required_vars = {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "WHATSAPP_TOKEN": WHATSAPP_TOKEN, 
        "WHATSAPP_PHONE_NUMBER_ID": WHATSAPP_PHONE_NUMBER_ID,
        "WHATSAPP_VERIFY_TOKEN": WHATSAPP_VERIFY_TOKEN
    }
    
    missing = [var for var, value in required_vars.items() if not value]
    if missing:
        print(f"❌ Error: Missing required environment variables: {missing}")
        print("Please check your .env file and ensure all variables are set.")
        return False
    else:
        print("✅ All required environment variables are configured")
        print(f"🤖 Using Google Gemini model: {MODEL_NAME}")
        print(f"🌟 AI Provider: {AI_PROVIDER}")
        return True

# Validate environment on startup
if not validate_environment():
    exit(1)

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Fallback functions if modules are missing
if not RAG_SERVICES_AVAILABLE:
    print("🔧 Creating fallback RAG services...")
    
    def transcription_detect_language(text):
        """Fallback language detection"""
        try:
            import langid
            lang, _ = langid.classify(text)
            return lang
        except ImportError:
            if any(char in text for char in 'देवनागरी'):
                return 'hi'
            elif any(char in text for char in 'मराठी'):
                return 'mr'
            else:
                return 'en'
    
    def build_rag_chain_from_documents(documents, google_api_key, model_name=None, **kwargs):
        """Fallback RAG chain builder"""
        print("⚠️ Using fallback RAG chain - limited functionality")
        return None
    
    def build_rag_chain_with_model_choice(pdf_file, txt_file, google_api_key, **kwargs):
        """Fallback RAG chain builder"""
        print("⚠️ Using fallback RAG chain - limited functionality")
        return None
    
    def process_scheme_query_with_retry(rag_chain, query, **kwargs):
        """Fallback query processor"""
        return ("Service temporarily unavailable. Please try again later.", "", "en", {"cache": "disabled"})
    
    def clear_query_cache():
        """Fallback cache clear"""
        print("Cache clearing not available")
    
    def get_model_options():
        """Fallback model options"""
        return {"gemini-1.5-flash": {"name": "Gemini 1.5 Flash", "description": "Default model"}}

if not TRANSCRIPTION_AVAILABLE:
    print("🔧 Creating fallback transcription...")
    
    def transcribe_audio(api_key, audio_bytes, method="fallback"):
        """Fallback transcription"""
        return (False, "Audio transcription not available. Please install required dependencies.")

def send_whatsapp_message(to: str, message: str) -> bool:
    """Send a WhatsApp message using the WhatsApp Business API"""
    try:
        headers = {
            'Authorization': f'Bearer {WHATSAPP_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message}
        }
        
        url = f"https://graph.facebook.com/v12.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            logging.info(f"WhatsApp message sent successfully to {to}")
            return True
        else:
            logging.error(f"Failed to send WhatsApp message: Status {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {str(e)}")
        return False

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
    "last_reset": datetime.now().date()
}

# Track processed message IDs
PROCESSED_MESSAGE_IDS = set()

def track_usage(action: str):
    """Simple usage tracking"""
    global USAGE_STATS
    
    today = datetime.now().date()
    if today != USAGE_STATS["last_reset"]:
        USAGE_STATS["daily_messages"] = 0
        USAGE_STATS["last_reset"] = today
    
    if action == "message_sent":
        USAGE_STATS["messages_sent"] += 1
        USAGE_STATS["daily_messages"] += 1
    elif action == "message_received":
        USAGE_STATS["messages_received"] += 1
    elif action == "api_call":
        USAGE_STATS["api_calls"] += 1

# API Status Check Functions
async def check_gemini_api_status():
    """Check Google Gemini API status and limits"""
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "message": "API key not configured",
            "available": False
        }
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        
        start_time = time.time()
        response = model.generate_content("Hello")
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "online",
            "response_time_ms": response_time,
            "model": MODEL_NAME,
            "available": True,
            "rate_limits": {
                "requests_per_minute": 15,
                "requests_per_day": 1500,
                "tokens_per_month": 1000000,
                "cost": "FREE"
            }
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "google-generativeai not installed",
            "available": False
        }
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return {
                "status": "rate_limited",
                "message": "Rate limit exceeded",
                "available": False
            }
        elif "api key" in error_msg.lower() or "invalid" in error_msg.lower():
            return {
                "status": "unauthorized", 
                "message": "Invalid API key",
                "available": False
            }
        else:
            return {
                "status": "error",
                "message": f"Connection failed: {error_msg}",
                "available": False
            }

class RedisManager:
    def __init__(self):
        self.redis_client = None
        self._connect()

    def _connect(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            print("✅ Redis connected successfully")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def is_available(self) -> bool:
        return self.redis_client is not None

    def set_rate_limit(self, key: str, expire_seconds: int = 60):
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

class QueryRequest(BaseModel):
    input_text: str
    model: str = "gemini-1.5-flash"
    enhanced_mode: bool = True
    voice_lang_pref: str = "auto"
    session_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up FastAPI application...")
    validate_environment()
    yield
    if redis_manager.is_available():
        redis_manager.redis_client.close()
    print("🛑 FastAPI application shutting down...")

app = FastAPI(title="SAMNEX AI - Google Gemini Edition", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def debug_gemini_request(query, model_choice="gemini-1.5-flash"):
    """Test Google Gemini API directly"""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_choice)
        
        print(f"🔍 Testing Google Gemini API directly...")
        print(f"   Model: {model_choice}")
        print(f"   Query: {query[:100]}...")
        
        response = model.generate_content(query)
        
        if response.text:
            print("✅ Direct Google Gemini API call successful!")
            return True, response.text
        else:
            return False, "No response text generated"
        
    except ImportError:
        return False, "google-generativeai not installed"
    except Exception as e:
        print(f"❌ Direct Google Gemini API call failed: {e}")
        return False, str(e)

def rebuild_rag_chain():
    """Rebuild RAG chain when knowledge base is updated"""
    global RAG_CHAIN
    try:
        if KNOWLEDGE_BASE["documents"] and RAG_SERVICES_AVAILABLE:
            print(f"🔄 Rebuilding RAG chain with Google Gemini: {MODEL_NAME}...")
            RAG_CHAIN = build_rag_chain_from_documents(
                KNOWLEDGE_BASE["documents"],
                google_api_key=GOOGLE_API_KEY,
                model_name=MODEL_NAME,
                enhanced_mode=True
            )
            KNOWLEDGE_BASE["last_updated"] = time.time()
            print(f"✅ RAG chain rebuilt successfully with {MODEL_NAME}")
        else:
            RAG_CHAIN = None
            print("⚠️ No documents available or RAG services not available")
    except Exception as e:
        print(f"❌ Error rebuilding RAG chain: {e}")
        import traceback
        traceback.print_exc()
        RAG_CHAIN = None

@app.get("/")
async def root():
    try:
        redis_status = redis_manager.is_available()
        kb_status = len(KNOWLEDGE_BASE["documents"]) if KNOWLEDGE_BASE["documents"] else 0
        return {
            "message": "CMRF AI Agent FastAPI backend is running with Google Gemini.", 
            "docs": "/docs", 
            "health": "/health/", 
            "api_status": "/api/status/comprehensive",
            "debug_documents": "/debug/documents",
            "redis_available": redis_status,
            "knowledge_base_documents": kb_status,
            "last_updated": KNOWLEDGE_BASE["last_updated"],
            "ai_provider": "Google Gemini",
            "model": MODEL_NAME,
            "cost": "FREE",
            "services_status": {
                "rag_services": RAG_SERVICES_AVAILABLE,
                "transcription": TRANSCRIPTION_AVAILABLE
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "redis_available": redis_manager.is_available(),
        "knowledge_base_documents": len(KNOWLEDGE_BASE["documents"]) if KNOWLEDGE_BASE["documents"] else 0,
        "rate_limit_seconds": 5,
        "timestamp": time.time(),
        "ai_provider": "Google Gemini",
        "model": MODEL_NAME,
        "cost": "FREE",
        "services_available": {
            "rag_services": RAG_SERVICES_AVAILABLE,
            "transcription": TRANSCRIPTION_AVAILABLE
        }
    }

@app.get("/api/status/gemini")
async def gemini_status_only():
    """Check only Google Gemini API status"""
    return await check_gemini_api_status()

@app.post("/upload/")
async def upload_knowledge_files(
    pdf_file: Optional[UploadFile] = File(None),
    txt_file: Optional[UploadFile] = File(None)
):
    """Upload PDF and/or TXT files to update the knowledge base"""
    try:
        if not pdf_file and not txt_file:
            return JSONResponse(
                status_code=400, 
                content={"error": "Please upload at least one PDF or TXT file"}
            )

        saved_files = []
        
        # Process PDF file
        if pdf_file:
            pdf_bytes = await pdf_file.read()
            os.makedirs("uploads", exist_ok=True)
            pdf_filename = os.path.join("uploads", f"{int(time.time())}_{pdf_file.filename}")
            with open(pdf_filename, "wb") as f:
                f.write(pdf_bytes)
                saved_files.append(pdf_filename)
            print(f"📄 PDF file saved: {pdf_filename}")
            
            # Read PDF content and add to knowledge base
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
                print(f"❌ Error reading PDF: {e}")

        # Process TXT file
        if txt_file:
            txt_bytes = await txt_file.read()
            os.makedirs("uploads", exist_ok=True)
            txt_filename = os.path.join("uploads", f"{int(time.time())}_{txt_file.filename}")
            with open(txt_filename, "wb") as f:
                f.write(txt_bytes)
                saved_files.append(txt_filename)
            print(f"📄 TXT file saved: {txt_filename}")
            
            # Read TXT content and add to knowledge base
            try:
                with open(txt_filename, "r", encoding="utf-8") as f:
                    text = f.read()
                    KNOWLEDGE_BASE["documents"].append({"filename": txt_filename, "content": text})
            except Exception as e:
                print(f"❌ Error reading TXT: {e}")

        # Rebuild RAG chain with new documents
        rebuild_rag_chain()

        return {
            "message": "Files uploaded successfully and knowledge base updated",
            "saved_files": saved_files,
            "total_documents": len(KNOWLEDGE_BASE["documents"]),
            "timestamp": time.time(),
            "ai_provider": "Google Gemini",
            "model": MODEL_NAME
        }

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query/")
async def query_knowledge_base(req: QueryRequest):
    """Query the knowledge base using uploaded documents"""
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
                    "error": "No knowledge base available. Please upload PDF/TXT files first.",
                    "upload_endpoint": "/upload/"
                }
            )

        # Generate response using AI and uploaded documents
        response = intelligent_rag_response(query)
        
        # Add validation here
        user_language = detect_language(query)
        
        # Validate response format
        if not validate_response_format(response, user_language):
            # Fix response format if validation fails
            response = fix_response_format(response, user_language)
            logging.info(f"Response format was fixed for language: {user_language}")
        
        return {
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "knowledge_base_size": len(KNOWLEDGE_BASE["documents"]),
            "ai_provider": "Google Gemini",
            "model": MODEL_NAME,
            "language_detected": user_language
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def intelligent_rag_response(query: str) -> str:
    """Enhanced response function with Google Gemini"""
    print(f"🤖 Processing query with Google Gemini: '{query}'")
    
    # Test language detection
    try:
        language = transcription_detect_language(query)
        print(f"🗣️ Detected language: {language}")
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}, defaulting to English")
        language = 'en'
    
    # Convert language codes to full names for validation
    language_map = {'hi': 'hindi', 'mr': 'marathi', 'en': 'english'}
    validation_language = language_map.get(language, 'english')
    
    # Handle greetings
    greetings = ['hi', 'hello', 'hey', 'namaste', 'नमस्ते', 'नमस्कार']
    if query.strip().lower() in greetings:
        if language == 'hi':
            greeting_response = "नमस्ते! मैं आपकी कैसे मदद कर सकती हूँ? 😊\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**"
        elif language == 'mr':
            greeting_response = "नमस्कार! मी तुमची कशी मदत करू शकते? 😊\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
        else:
            greeting_response = "Hello! How can I help you today? 😊\n\n**For more details, please contact the 104/102 helpline numbers.**"
        
        return greeting_response
    
    # Check if services are available
    if not RAG_SERVICES_AVAILABLE:
        error_response = "🔧 AI services are being set up. Please update rag_services.py and restart the application."
        return fix_response_format(error_response, validation_language)
    
    # Check knowledge base
    if not KNOWLEDGE_BASE["documents"]:
        error_messages = {
            'hi': "📚 ज्ञान संग्रह उपलब्ध नहीं है। कृपया प्रशासक से संपर्क करें।",
            'mr': "📚 ज्ञान संग्रह उपलब्ध नाही. कृपया प्रशासकाशी संपर्क साधा.",
            'en': "📚 Knowledge base not available. Please contact administrator."
        }
        error_response = error_messages.get(language, error_messages['en'])
        return fix_response_format(error_response, validation_language)
    
    # Test Google Gemini API
    print("🔍 Testing Google Gemini API connection...")
    api_test_success, api_result = debug_gemini_request(query[:100])
    
    if not api_test_success:
        print(f"❌ Google Gemini API test failed: {api_result}")
        error_messages = {
            'hi': "तकनीकी समस्या हुई है। कृपया बाद में कोशिश करें।",
            'mr': "तांत्रिक समस्या आली आहे. कृपया नंतर प्रयत्न करा.",
            'en': "Technical issue occurred. Please try again later."
        }
        error_response = error_messages.get(language, error_messages['en'])
        return fix_response_format(error_response, validation_language)
    
    print("✅ Google Gemini API test successful")
    
    # Build/rebuild RAG chain if needed
    global RAG_CHAIN
    if RAG_CHAIN is None:
        try:
            print("🔄 Building RAG chain...")
            rebuild_rag_chain()
            print("✅ RAG chain built successfully")
        except Exception as e:
            print(f"❌ Error building RAG chain: {e}")
            error_messages = {
                'hi': "सिस्टम तैयार नहीं हो सका। प्रशासक से संपर्क करें।",
                'mr': "सिस्टम तयार होऊ शकला नाही. प्रशासकाशी संपर्क साधा.",
                'en': "System could not be prepared. Contact administrator."
            }
            error_response = error_messages.get(language, error_messages['en'])
            return fix_response_format(error_response, validation_language)
    
    # Process the query
    try:
        print(f"🔄 Processing query with Google Gemini RAG chain...")
        
        # Use RAG chain
        result = process_scheme_query_with_retry(RAG_CHAIN, query)
        result_text = result[0] if isinstance(result, tuple) else str(result)
        
        print(f"✅ Google Gemini RAG query successful: {result_text[:100]}...")
        
        # Clean up response
        result_text = result_text.strip()
        if not result_text:
            error_messages = {
                'hi': "कोई जवाब नहीं मिला। कृपया अपना सवाल दूसरे तरीके से पूछें।",
                'mr': "कोणताही उत्तर मिळाला नाही. कृपया आपला प्रश्न वेगळ्या पद्धतीने विचारा.",
                'en': "No answer found. Please rephrase your question."
            }
            error_response = error_messages.get(language, error_messages['en'])
            return fix_response_format(error_response, validation_language)
        
        # Convert markdown bold to WhatsApp bold
        result_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', result_text)
        result_text = re.sub(r'(\*[^*]+\*)\n([^\n*])', r'\1\n\n\2', result_text)
        
        # Validate and fix response format before returning
        if not validate_response_format(result_text, validation_language):
            result_text = fix_response_format(result_text, validation_language)
            logging.info(f"Fixed response format for language: {validation_language}")
        
        return result_text
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        
        error_str = str(e).lower()
        if "quota" in error_str or "limit" in error_str:
            error_messages = {
                'hi': "दैनिक सीमा समाप्त। कृपया कल पुनः प्रयास करें।",
                'mr': "दैनंदिन मर्यादा संपली. कृपया उद्या पुन्हा प्रयत्न करा.",
                'en': "Daily limit reached. Please try again tomorrow."
            }
        elif "safety" in error_str:
            error_messages = {
                'hi': "सुरक्षा कारणों से जवाब नहीं दे सकते। कृपया सवाल बदलें।",
                'mr': "सुरक्षितता कारणांमुळे उत्तर देऊ शकत नाही. कृपया प्रश्न बदला.",
                'en': "Cannot respond due to safety guidelines. Please rephrase."
            }
        else:
            error_messages = {
                'hi': "तकनीकी समस्या। कृपया दोबारा कोशिश करें।",
                'mr': "तांत्रिक समस्या. कृपया पुन्हा प्रयत्न करा.",
                'en': "Technical issue. Please try again."
            }
        
        error_response = error_messages.get(language, error_messages['en'])
        return fix_response_format(error_response, validation_language)

@app.post("/voice-query/")
async def voice_query(audio_file: UploadFile = File(...)):
    """
    Process voice queries through audio transcription with multi-language support
    Supports Hindi, Marathi, and English audio input
    """
    try:
        # Read audio bytes
        audio_bytes = await audio_file.read()
        
        print(f"🎤 Received audio file: {audio_file.filename}, Size: {len(audio_bytes)} bytes")
        
        # Transcribe audio using improved transcription
        success, transcription_or_error = transcribe_audio(
            GOOGLE_API_KEY, 
            audio_bytes, 
            method="local"  # Use local Whisper for best multi-language support
        )
        
        if not success:
            return JSONResponse(
                status_code=400,
                content={
                    "error": transcription_or_error,
                    "supported_languages": list(SUPPORTED_LANGUAGES.values()),
                    "message": "Please send your message as text or try speaking more clearly"
                }
            )
        
        transcription = transcription_or_error
        print(f"✅ Transcription successful: '{transcription}'")
        
        # Detect language of transcribed text
        try:
            from core.transcription import detect_audio_language
            lang_code, lang_name, confidence = detect_audio_language(transcription)
            print(f"🗣️ Audio language detected: {lang_name} ({lang_code}) - Confidence: {confidence:.2f}")
        except ImportError:
            lang_code, lang_name, confidence = "en", "English", 0.8
        
        # Process transcription as a text query
        print(f"🤖 Processing transcribed text with Google Gemini...")
        answer = intelligent_rag_response(transcription)
        
        return {
            "transcription": {
                "text": transcription,
                "detected_language": {
                    "code": lang_code,
                    "name": lang_name,
                    "confidence": confidence
                }
            },
            "query_processed": transcription,
            "response": answer,
            "timestamp": time.time(),
            "ai_provider": "Google Gemini",
            "transcription_method": "local_whisper",
            "supported_languages": list(SUPPORTED_LANGUAGES.values())
        }
        
    except Exception as e:
        print(f"❌ Voice query error: {e}")
        traceback.print_exc()
        
        # Return language-specific error message
        error_responses = {
            "en": "I couldn't process your audio. Please try sending a text message instead.",
            "hi": "मैं आपकी आवाज़ को समझ नहीं सकी। कृपया टेक्स्ट संदेश भेजने का प्रयास करें।",
            "mr": "मी तुमचा आवाज समजू शकलो नाही. कृपया मजकूर संदेश पाठवण्याचा प्रयत्न करा."
        }
        
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "message": error_responses["en"],
                "message_hi": error_responses["hi"], 
                "message_mr": error_responses["mr"],
                "supported_languages": list(SUPPORTED_LANGUAGES.values())
            }
        )

# Also add this test endpoint for audio transcription testing

@app.post("/debug/test-audio-transcription")
async def test_audio_transcription(audio_file: UploadFile = File(...)):
    """
    Test endpoint for audio transcription debugging
    """
    try:
        audio_bytes = await audio_file.read()
        
        print(f"🧪 Testing audio transcription...")
        print(f"   File: {audio_file.filename}")
        print(f"   Size: {len(audio_bytes)} bytes")
        
        # Test different transcription methods
        methods = ["local", "fallback"]
        results = {}
        
        for method in methods:
            try:
                success, result = transcribe_audio(GOOGLE_API_KEY, audio_bytes, method=method)
                results[method] = {
                    "success": success,
                    "result": result[:200] if success else result
                }
                
                # If successful, also test language detection
                if success:
                    try:
                        from core.transcription import detect_audio_language
                        lang_code, lang_name, confidence = detect_audio_language(result)
                        results[method]["language_detection"] = {
                            "code": lang_code,
                            "name": lang_name,
                            "confidence": confidence
                        }
                    except:
                        results[method]["language_detection"] = "Detection failed"
                        
            except Exception as e:
                results[method] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "file_info": {
                "filename": audio_file.filename,
                "size_bytes": len(audio_bytes),
                "content_type": audio_file.content_type
            },
            "transcription_tests": results,
            "supported_languages": list(SUPPORTED_LANGUAGES.values()),
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Audio transcription test failed"}
        )

# WhatsApp webhook endpoints
@app.get("/webhook")
async def verify_whatsapp(request: Request):
    """Verification endpoint for Meta WhatsApp Cloud API"""
    try:
        params = dict(request.query_params)
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")
        
        print(f"🔍 Webhook verification: mode={mode}, token={token}")
        
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            print("✅ VERIFICATION SUCCESS!")
            return int(challenge)
        else:
            print("❌ VERIFICATION FAILED!")
            return JSONResponse(status_code=403, content={"error": "Verification failed"})
            
    except Exception as e:
        print(f"💥 Verification error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Handle incoming WhatsApp messages"""
    global PROCESSED_MESSAGE_IDS
    
    try:
        data = await request.json()
        
        # Basic message extraction
        if "entry" not in data or not data["entry"]:
            return {"status": "ok"}
        
        entry = data["entry"][0]
        if "changes" not in entry or not entry["changes"]:
            return {"status": "ok"}
        
        change = entry["changes"][0]
        value = change.get("value", {})
        
        # Ignore status updates
        if "statuses" in value:
            return {"status": "ok"}
        
        # Handle messages
        if "messages" not in value or not value["messages"]:
            return {"status": "ok"}
        
        message_obj = value["messages"][0]
        message_type = message_obj.get("type", "")
        user_number = message_obj.get("from", "")
        message_id = message_obj.get("id", "")
        
        print(f"📝 MESSAGE DETAILS:")
        print(f"   Type: {message_type}")
        print(f"   From: {user_number}")
        print(f"   Message ID: {message_id}")
        
        # Check for duplicates
        if message_id in PROCESSED_MESSAGE_IDS:
            print(f"🔄 DUPLICATE MESSAGE IGNORED: {message_id}")
            return {"status": "duplicate_ignored"}
        
        # Prevent responding to our own messages
        if user_number == WHATSAPP_PHONE_NUMBER_ID:
            print("🔄 IGNORING MESSAGE FROM OUR OWN BOT")
            return {"status": "ignored_own_message"}
        
        if message_type == "text":
            user_msg = message_obj.get("text", {}).get("body", "").strip()
            
            if user_msg:
                print(f"💬 Received: '{user_msg}' from {user_number}")
                
                # Track usage
                track_usage("message_received")
                PROCESSED_MESSAGE_IDS.add(message_id)
                
                # Keep only recent 1000 message IDs
                if len(PROCESSED_MESSAGE_IDS) > 1000:
                    old_ids = list(PROCESSED_MESSAGE_IDS)
                    PROCESSED_MESSAGE_IDS = set(old_ids[-500:])
                
                # Rate limiting
                rate_limit_key = f"whatsapp_{user_number}"
                if redis_manager.check_rate_limit(rate_limit_key):
                    print(f"⏱️ Rate limited user: {user_number}")
                    return {"status": "rate_limited"}
                
                redis_manager.set_rate_limit(rate_limit_key, 5)
                
                # Generate response
                print(f"🤖 Processing message with Google Gemini: '{user_msg}'")
                response = intelligent_rag_response(user_msg)
                
                # Validate response
                if not response or len(response.strip()) == 0:
                    response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
                # Send response with retry logic
                print(f"📤 Sending response to: {user_number}")
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    success = send_whatsapp_message(user_number, response)
                    if success:
                        track_usage("message_sent")
                        break
                    else:
                        print(f"❌ Send attempt {attempt + 1} failed")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                
                print(f"📤 Final send result: {'SUCCESS' if success else 'FAILED'}")
                
                return {
                    "status": "processed",
                    "message_id": message_id,
                    "send_success": success,
                    "response_length": len(response),
                    "ai_provider": "Google Gemini"
                }
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"💥 Webhook error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/whatsapp/test")
async def test_whatsapp_message(request: Request):
    """Test endpoint to send a WhatsApp message"""
    try:
        data = await request.json()
        phone_number = data.get("phone_number")
        message = data.get("message", "Test message from CMRF AI with Google Gemini")
        
        if not phone_number:
            return JSONResponse(status_code=400, content={"error": "phone_number is required"})
        
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number.lstrip('+')
        
        success = send_whatsapp_message(phone_number, message)
        
        if success:
            track_usage("message_sent")
        
        return {
            "success": success,
            "phone_number": phone_number,
            "message": message,
            "timestamp": time.time(),
            "ai_provider": "Google Gemini"
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/debug/test-query")
async def test_query_debug(request: Request):
    """Debug endpoint to test query processing"""
    try:
        data = await request.json()
        test_query = data.get("query", "Hello")
        
        print(f"🧪 Testing query: {test_query}")
        
        # Test 1: Direct Google Gemini API
        api_success, api_result = debug_gemini_request(test_query)
        
        # Test 2: Language detection
        try:
            language = transcription_detect_language(test_query)
            lang_success = True
        except Exception as e:
            language = str(e)
            lang_success = False
        
        # Test 3: Knowledge base check
        kb_available = bool(KNOWLEDGE_BASE["documents"])
        
        # Test 4: Full RAG response if possible
        try:
            rag_response = intelligent_rag_response(test_query)
            rag_success = True
        except Exception as e:
            rag_response = str(e)
            rag_success = False
        
        return {
            "test_query": test_query,
            "google_gemini_api_test": {
                "success": api_success,
                "result": api_result[:200] if api_success else api_result
            },
            "language_detection": {
                "success": lang_success,
                "detected": language
            },
            "knowledge_base": {
                "available": kb_available,
                "document_count": len(KNOWLEDGE_BASE["documents"])
            },
            "rag_response": {
                "success": rag_success,
                "response": rag_response[:200] if rag_success else rag_response
            },
            "timestamp": time.time(),
            "ai_provider": "Google Gemini",
            "model": MODEL_NAME
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Debug test failed"}
        )

@app.get("/usage/stats")
async def get_usage_stats():
    """Get current usage statistics"""
    uptime_hours = (time.time() - USAGE_STATS["start_time"]) / 3600
    
    return {
        "current_usage": {
            "messages_sent_total": USAGE_STATS["messages_sent"],
            "messages_received_total": USAGE_STATS["messages_received"], 
            "messages_sent_today": USAGE_STATS["daily_messages"],
            "gemini_api_calls": USAGE_STATS["api_calls"]
        },
        "estimated_limits": {
            "whatsapp_daily_test": "~100-500 messages",
            "whatsapp_rate": "~10 per minute",
            "gemini_per_day": "1500 requests (FREE)",
            "gemini_per_minute": "15 requests"
        },
        "warnings": {
            "daily_messages": "⚠️ High usage" if USAGE_STATS["daily_messages"] > 50 else "✅ Normal",
            "total_messages": "⚠️ Consider production" if USAGE_STATS["messages_sent"] > 200 else "✅ Test mode OK"
        },
        "uptime_hours": round(uptime_hours, 2),
        "status": "test_mode",
        "rate_limit_setting": "5 seconds",
        "ai_provider": "Google Gemini",
        "model": MODEL_NAME,
        "cost": "FREE"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server with Google Gemini...")
    print("📋 Setup Status:")
    print(f"   ✅ Environment variables: OK")
    print(f"   {'✅' if RAG_SERVICES_AVAILABLE else '❌'} RAG Services: {'Available' if RAG_SERVICES_AVAILABLE else 'Missing - update core/rag_services.py'}")
    print(f"   {'✅' if TRANSCRIPTION_AVAILABLE else '❌'} Transcription: {'Available' if TRANSCRIPTION_AVAILABLE else 'Missing - create core/transcription.py'}")
    print(f"   🤖 Model: {MODEL_NAME}")
    print(f"   💰 Cost: FREE")
    print("")
    print("🔧 Next steps if RAG services missing:")
    print("   1. Replace core/rag_services.py with Google Gemini version")
    print("   2. Install: pip install google-generativeai langchain langchain-community")
    print("   3. Upload documents via /upload/ endpoint")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8080)