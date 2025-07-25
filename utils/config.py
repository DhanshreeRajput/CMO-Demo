import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
AI_PROVIDER = os.getenv("AI_PROVIDER", "google")

# WhatsApp Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID") 
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Optional Settings
REDIS_CACHE_ENABLED = os.getenv("REDIS_CACHE_ENABLED", "true").lower() == "true"
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 5))

def validate_google_config():
    """Validate Google Gemini configuration"""
    if not GOOGLE_API_KEY:
        print("❌ Warning: GOOGLE_API_KEY not set")
        return False
    
    if not GOOGLE_API_KEY.startswith("AIza"):
        print("❌ Warning: Invalid Google API key format (should start with 'AIza')")
        return False
    
    print(f"✅ Google Gemini Model: {MODEL_NAME}")
    print(f"✅ AI Provider: {AI_PROVIDER}")
    return True

def validate_whatsapp_config():
    """Validate WhatsApp configuration"""
    required_vars = [WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN]
    missing = [var for var in required_vars if not var]
    
    if missing:
        print(f"❌ Missing WhatsApp config: {len(missing)} variables")
        return False
    
    print("✅ WhatsApp configuration valid")
    return True

def validate_all_config():
    """Validate all configuration"""
    google_valid = validate_google_config()
    wa_valid = validate_whatsapp_config()
    
    return google_valid and wa_valid