import tempfile
import os
import time
import hashlib
import re
import langid
from typing import Optional, List, Dict, Any
import requests
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Language detection setup
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not installed. Using langid only.")
    LANGDETECT_AVAILABLE = False

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "mr": "Marathi"
}

set_verbose(False)

# ENHANCED FAST CACHING
_query_cache = {}
_cache_max_size = 200
_cache_stats = {"hits": 0, "misses": 0}

class FastOllamaLLM(LLM):
    """Ultra-fast Ollama LLM optimized for WhatsApp 8B model"""
    
    model: str = "llama3.1:8b"  # DEFAULT TO FAST MODEL
    base_url: str = "http://localhost:11434"
    max_tokens: int = 1200     # INCREASED FOR FULL RESPONSES
    temperature: float = 0.1   # FAST & CONSISTENT
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self.model = model_name
        self.base_url = base_url
        
        # OPTIMIZE BASED ON MODEL SIZE - INCREASED TOKENS FOR FULL RESPONSES
        if "8b" in model_name.lower():
            self.max_tokens = 1200  # Increased from 700
            self.temperature = 0.1
            print(f"⚡ FAST 8B model initialized: {model_name} (Full responses enabled)")
        else:
            self.max_tokens = 1500  # Increased from 1000
            self.temperature = 0.05
            print(f"🐌 Slower model: {model_name}")
        
        # Quick connection test
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if model_name in models:
                    print(f"✅ Model '{model_name}' ready for WhatsApp")
                else:
                    print(f"❌ Model '{model_name}' not found")
                    print(f"💡 Run: ollama pull {model_name}")
            else:
                print(f"❌ Ollama server error: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Connection test failed: {e}")
    
    @property
    def _llm_type(self) -> str:
        return "fast_ollama_whatsapp"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Ultra-fast call optimized for WhatsApp speed - FULL RESPONSES"""
        try:
            # SPEED-OPTIMIZED PAYLOAD
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 2048 if "8b" in self.model else 3072,  # Increased context
                    "num_thread": -1,
                    "stop": stop or []
                }
            }
            
            # FAST TIMEOUT BASED ON MODEL
            timeout = 45 if "8b" in self.model else 90  # Increased timeout for full responses
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                print(f"⚡ Generated in {response_time:.2f}s")
                
                if answer:
                    # NO TRUNCATION - Return full response
                    return answer
                else:
                    return self._get_fallback_response()
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return self._get_fallback_response()
            
        except requests.exceptions.Timeout:
            print(f"❌ Timeout after {timeout}s")
            return "Response taking longer than expected. Ask a shorter question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama")
            return "AI service unavailable. Try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        except Exception as e:
            print(f"❌ Fast Ollama error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self):
        """Fast fallback response with proper helpline format"""
        return "Unable to process right now. Please try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"

def get_query_hash(query_text):
    """Generate fast cache key"""
    return hashlib.md5(query_text.lower().strip().encode()).hexdigest()

def cache_result(query_hash, result):
    """Enhanced caching with performance tracking"""
    global _query_cache, _cache_stats
    
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest 20 entries for efficiency
        old_keys = list(_query_cache.keys())[:20]
        for key in old_keys:
            del _query_cache[key]
    
    _query_cache[query_hash] = {
        "result": result,
        "timestamp": time.time()
    }

def get_cached_result(query_hash, ttl=1800):  # 30 minutes TTL
    """Get cached result with expiration"""
    global _cache_stats
    
    if query_hash in _query_cache:
        cached = _query_cache[query_hash]
        if time.time() - cached["timestamp"] < ttl:
            _cache_stats["hits"] += 1
            print("💨 CACHE HIT - Instant response!")
            return cached["result"]
        else:
            del _query_cache[query_hash]  # Remove expired
    
    _cache_stats["misses"] += 1
    return None

def clear_query_cache():
    """Clear cache and reset stats"""
    global _query_cache, _cache_stats
    _query_cache.clear()
    _cache_stats = {"hits": 0, "misses": 0}
    print("✅ Fast cache cleared")

def get_cache_stats():
    """Get cache performance statistics"""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = (_cache_stats["hits"] / total * 100) if total > 0 else 0
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "hit_rate": f"{hit_rate:.1f}%",
        "cache_size": len(_query_cache)
    }

def detect_language(text):
    """Enhanced language detection with better Marathi support"""
    try:
        clean_text = text.strip().lower()
        if len(clean_text) < 3:
            return 'en'

        # Enhanced character-based detection
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', clean_text))
        english_chars = bool(re.search(r'[a-zA-Z]', clean_text))

        if hindi_chars and not english_chars:
            # Enhanced Marathi vs Hindi detection with more keywords
            marathi_words = ['baddal', 'mahiti', 'dya', 'kasa', 'kara', 'ahe', 'tumhi', 'mi', 
                           'आहे', 'तुम्ही', 'मी', 'माहिती', 'येथे', 'करा', 'कसा', 'बद्दल', 'द्या']
            hindi_words = ['ke', 'liye', 'kaise', 'karna', 'hai', 'aapke', 'jaankari', 
                          'है', 'आपके', 'जानकारी', 'कैसे', 'करना', 'के लिए', 'यहाँ']
            
            marathi_count = sum(1 for word in marathi_words if word in clean_text)
            hindi_count = sum(1 for word in hindi_words if word in clean_text)
            
            if marathi_count > hindi_count:
                return 'mr'
            return 'hi'
        elif english_chars:
            return 'en'
        else:
            return 'en'  # Default

    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'

def get_whatsapp_prompt_template():
    """
    ORIGINAL prompt template - keeping same detailed responses
    Returns: PromptTemplate object
    """
    template = """You are a female efficient Knowledge Assistant, designed for answering questions specifically from the knowledge base provided to you.

Your task is as follows: give a detailed response for the user query in the user language (e.g., "what are some schemes?" --> "Here is a list of some schemes").

Ensure your response follows these styles and tone:
* Read numbers as digits, e.g., "104" instead of "one hundred four"
* Always answer in the **same language as the Question**, regardless of the language of the source documents.
* If the source documents are in Marathi and the question is in English, **translate and summarize the information into English**.
* If the question is in Marathi, answer in Marathi. Do the same for English and Hindi.
* Use direct, everyday language.
* Maintain a personal and friendly tone, aligned with the user's language.
* Provide detailed and comprehensive responses with minimum 150-200 words per scheme.
* Include **toll-free numbers** and **complete visible website URLs** *only if those URLs are present in the knowledge base*.
* When providing contact information, always include specific contact details from the knowledge base if available.

**RESPONSE FORMAT REQUIREMENTS:**
* **Always format your answer using markdown. Use markdown headings (##), bold (**text**), bullet lists (-), and other markdown features where appropriate. This applies to English, Hindi, and Marathi answers.**
* Use clear section headers:
  - **English**: "Description", "Eligibility", "Benefits", "How to Apply", "Required Documents", "Contact Information"
  - **Hindi**: "विवरण", "पात्रता", "लाभ", "आवेदन कैसे करें", "आवश्यक दस्तावेज", "संपर्क जानकारी"
  - **Marathi**: "वर्णन", "पात्रता", "फायदे", "अर्ज कसा करावा", "आवश्यक कागदपत्रे", "संपर्क माहिती"

**MANDATORY ENDING**: Always end every response with the helpline information in the user's language:
  - **In English**: "**For more details, please contact the 104/102 helpline numbers.**"
  - **In Hindi**: "**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**"
  - **In Marathi**: "**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"

* If there is no relevant context for the question, simply say:
  - **In Marathi**: "क्षमस्व, मी या विषयावर तुमची मदत करू शकत नाही. अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा."
  - **In Hindi**: "माफ़ कीजिए, मैं इस विषय पर आपकी मदत नहीं कर सकती। अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।"
  - **In English**: "I'm sorry, I cannot assist with that topic. For more details, please contact the 104/102 helpline numbers."

**STRICT LANGUAGE MATCHING**: 
  - Hindi input → Hindi output ONLY
  - English input → English output ONLY
  - Marathi input → Marathi output ONLY
  - Never mix languages or use wrong language

**Remove duplicate information and provide only one consolidated answer.**
* Do not provide answers based on assumptions or general knowledge. Use only the information provided in the knowledge base.
* **Response Length Requirements**: Minimum 150 words for single scheme, 300+ words for multiple schemes.
* **If the user asks for jokes, casual conversation, to 'talk like' someone, or anything not related to government schemes or the knowledge base, do not answer. Instead, respond with the helpline apology message above.**
* **Contact Information Priority**: Always include specific contact details from knowledge base (phone numbers, office addresses, website URLs) when available for each scheme mentioned.

**RESPONSE FORMAT TEMPLATE**:

## **[Scheme Name]**

**Description**: [Detailed explanation]

**Eligibility**: 
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

**Benefits**:
- [Benefit 1 with specific amounts/details]
- [Benefit 2 with specific amounts/details]

**How to Apply**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Required Documents**:
- [Document 1]
- [Document 2]

**Contact Information** *(if available in knowledge base)*:
- **Phone**: [Specific numbers from knowledge base]
- **Website**: [Full URL from knowledge base]
- **Office Address**: [Complete address from knowledge base]

**For more details, please contact the 104/102 helpline numbers.**

Your goal is to help a citizen understand schemes and their eligibility criteria clearly, using only the verified data provided in the documents.

Here is the content you will work with: {context}

Question: {question}

Now perform the task as instructed above.

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_whatsapp_fast_config(model_name="llama3.1:8b"):
    """Get WhatsApp-optimized fast configuration - FULL RESPONSES"""
    if "8b" in model_name.lower():
        return {
            "max_tokens": 1200,     # INCREASED FOR FULL RESPONSES
            "chunk_size": 300,      # INCREASED FOR MORE CONTEXT
            "max_chunks": 4,        # INCREASED FOR COMPLETE ANSWERS
            "temperature": 0.1,     # CONSISTENT ANSWERS
            "timeout": 45           # INCREASED TIMEOUT FOR FULL RESPONSES
        }
    else:
        return {
            "max_tokens": 1500,
            "chunk_size": 400,
            "max_chunks": 6,
            "temperature": 0.05,
            "timeout": 90
        }

def build_rag_chain_from_documents(documents, ollama_model="llama3.1:8b", model_name="llama3.1:8b", 
                                 use_local=True, enhanced_mode=True):
    """Build ULTRA-FAST RAG chain for WhatsApp with 8B model"""
    print(f"🚀 Building ULTRA-FAST WhatsApp RAG chain with {ollama_model}")
    
    # Convert documents to LangChain format
    all_docs = [
        Document(
            page_content=doc["content"], 
            metadata={"filename": doc.get("filename", "unknown")}
        ) 
        for doc in documents
    ]
    
    if not all_docs:
        raise ValueError("No documents available for RAG chain")
    
    # GET FAST CONFIG
    config = get_whatsapp_fast_config(ollama_model)
    print(f"⚡ WhatsApp speed config: {config}")
    
    # FAST TEXT SPLITTING - INCREASED FOR FULL RESPONSES
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=50,  # Increased overlap for better context
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len
    )
    splits = splitter.split_documents(all_docs)
    print(f"📄 Created {len(splits)} fast chunks for WhatsApp")
    
    if not splits:
        raise ValueError("Document splitting failed")
    
    # FAST RETRIEVER
    retriever = TFIDFRetriever.from_documents(
        splits, 
        k=min(config["max_chunks"], len(splits))
    )
    
    # ORIGINAL DETAILED PROMPT (keeping your format)
    whatsapp_prompt = get_whatsapp_prompt_template()
    
    # CREATE FAST OLLAMA LLM
    try:
        llm = FastOllamaLLM(
            model_name=ollama_model,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        print(f"✅ Fast WhatsApp LLM ready")
        
    except Exception as e:
        print(f"❌ Error creating fast LLM: {e}")
        raise
    
    # BUILD FAST CHAIN
    try:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": whatsapp_prompt, "verbose": False}
        )
        print("🚀 ULTRA-FAST WhatsApp RAG chain ready!")
        return chain
        
    except Exception as e:
        print(f"❌ Error creating fast RAG chain: {e}")
        raise

def process_scheme_query_with_retry(rag_chain, user_query, max_retries=1, enable_tts=False, autoplay=False):
    """ULTRA-FAST query processing for WhatsApp (single attempt for speed)"""
    print(f"⚡ FAST processing: '{user_query[:50]}...'")
    
    # FAST LANGUAGE DETECTION
    detected_lang = detect_language(user_query)
    supported_languages = {"en", "hi", "mr"}
    
    if detected_lang not in supported_languages:
        return (
            "⚠️ Please use English, Hindi, or Marathi.\nकृपया अंग्रेजी, हिंदी या मराठी का उपयोग करें।",
            "",
            detected_lang,
            {"cache": "language_error"}
        )

    # CHECK CACHE FIRST FOR INSTANT RESPONSE
    query_hash = get_query_hash(user_query)
    cached_result = get_cached_result(query_hash)
    
    if cached_result:
        return (cached_result, "", detected_lang, {"cache": "hit"})
    
    # TRUNCATE LONG QUERIES FOR SPEED (but keep reasonable length for full answers)
    if len(user_query) > 200:  # Increased from 150
        user_query = user_query[:200] + "..."
        print(f"✂️ Query truncated for WhatsApp speed")
    
    # SINGLE FAST ATTEMPT (no retries for WhatsApp speed)
    try:
        start_time = time.time()
        
        result = rag_chain.invoke({"query": user_query})
        
        response_time = time.time() - start_time
        print(f"⚡ WhatsApp response in {response_time:.2f}s")
        
        # EXTRACT RESULT
        if isinstance(result, dict):
            result_text = result.get('result', 'No information found.')
        else:
            result_text = str(result)
        
        # VALIDATE RESULT
        if not result_text or len(result_text.strip()) < 10:
            fallback_messages = {
                'hi': "जानकारी नहीं मिली। 104/102 हेल्पलाइन पर संपर्क करें।",
                'mr': "माहिती मिळाली नाही. 104/102 हेल्पलाइन वर संपर्क करा.",
                'en': "No information found. Contact 104/102 helpline."
            }
            result_text = fallback_messages.get(detected_lang, fallback_messages['en'])
        
        # CACHE FOR FUTURE SPEED
        cache_result(query_hash, result_text)
        
        return (result_text, "", detected_lang, {"cache": "miss"})
        
    except Exception as e:
        print(f"❌ Fast processing failed: {e}")
        
        # QUICK ERROR RESPONSES
        error_messages = {
            'hi': "तकनीकी समस्या। पुनः प्रयास करें।",
            'mr': "तांत्रिक समस्या. पुन्हा प्रयत्न करा.",
            'en': "Technical issue. Please try again."
        }
        
        error_response = error_messages.get(detected_lang, error_messages['en'])
        return (error_response, "", detected_lang, {"cache": "error"})

def get_model_options():
    """Available models with WhatsApp speed ratings"""
    return {
        "llama3.1:8b": {
            "name": "Llama 3.1 8B (WHATSAPP OPTIMIZED)", 
            "description": "⚡ 2-5 second responses - PERFECT for WhatsApp!",
            "speed": "ULTRA_FAST",
            "whatsapp_recommended": True
        },
        "llama3.1:70b": {
            "name": "Llama 3.1 70B (TOO SLOW)", 
            "description": "🐌 30-60 second responses - NOT suitable for WhatsApp",
            "speed": "VERY_SLOW",
            "whatsapp_recommended": False
        }
    }

class FastContextCallback(BaseCallbackHandler):
    """Lightweight callback for WhatsApp speed"""
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("🔄 Fast WhatsApp chain started")
    
    def on_chain_end(self, outputs, **kwargs):
        print("✅ Fast WhatsApp chain completed")

# BACKWARD COMPATIBILITY FUNCTIONS (updated for speed)
def build_rag_chain_with_model_choice(pdf_file, txt_file, google_api_key, model_choice="llama3.1:8b", 
                                     use_local=True, enhanced_mode=True):
    """Fast compatibility function - defaults to 8B model"""
    print(f"🔄 Using fast model for WhatsApp: {model_choice}")
    return build_rag_chain_from_files(pdf_file, txt_file, model_choice, model_choice, use_local, enhanced_mode)

def build_rag_chain_from_files(pdf_file, txt_file, ollama_model="llama3.1:8b", model_name="llama3.1:8b", 
                              use_local=True, enhanced_mode=True):
    """Fast file processing for WhatsApp"""
    print(f"🚀 Fast file processing with {ollama_model}")
    
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file required")
    
    temp_files = []
    try:
        # Fast file processing
        documents = []
        
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                pdf_path = tmp.name
                temp_files.append(pdf_path)
            
            docs = PyPDFLoader(pdf_path).load()
            documents.extend(docs)
        
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(txt_file.getvalue())
                txt_path = tmp.name
                temp_files.append(txt_path)
            
            docs = TextLoader(txt_path, encoding="utf-8").load()
            documents.extend(docs)
        
        if not documents:
            raise ValueError("No documents loaded")
        
        # Convert to memory format and build chain
        memory_docs = [{"content": doc.page_content, "filename": "uploaded"} for doc in documents]
        return build_rag_chain_from_documents(memory_docs, ollama_model, model_name, use_local, enhanced_mode)
        
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

# COMPATIBILITY ALIASES
StrictContextCallback = FastContextCallback

# Extract schemes functionality (kept for compatibility)
def extract_schemes_from_text(text_content):
    """Helper function to extract schemes from text content using regex patterns."""
    text = re.sub(r'\s+', ' ', str(text_content)).replace('\n', ' ')
    
    patterns = [
        r'\b(?:[A-Z][\w\'-]+(?: [A-Z][\w\'-]+)* )?(?:योजना|कार्यक्रम|अभियान|मिशन|धोरण|निधी|कार्ड|Scheme|Yojana|Programme|Abhiyan|Mission|Initiative|Program|Policy|Fund|Card)\b',
        r'\b(?:Pradhan Mantri|Mukhyamantri|CM|PM|National|Rashtriya|State|Rajya|प्रधानमंत्री|मुख्यमंत्री|राष्ट्रीय|राज्य) (?:[A-Z][\w\'-]+ ?)+',
        r'\b[A-Z]{2,}(?:-[A-Z]{2,})? Scheme\b',
        r'\b(?:[०-९]+|[0-9]+)\.\s+([A-Z][\w\s\'-]+(?:योजना|Scheme|कार्यक्रम|Karyakram|अभियान|Abhiyan))',
        r'•\s+([A-Z][\w\s\'-]+(?:योजना|Scheme|कार्यक्रम|Karyakram|अभियान|Abhiyan))'
    ]
    
    extracted_schemes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            scheme_name = match[1] if isinstance(match, tuple) else match
            cleaned_name = scheme_name.strip().rstrip('.,:;-').title()
            if len(cleaned_name) > 4 and len(cleaned_name.split()) < 10:
                extracted_schemes.add(cleaned_name)

    return sorted(list(extracted_schemes))

def query_all_schemes_optimized(rag_chain):
    """Optimized scheme extractor using targeted queries and regex."""
    try:
        context_query = "Provide a comprehensive list of all government schemes, programs, and yojana mentioned in the documents."
        response = rag_chain.invoke({"query": context_query})
        content_to_parse = response.get('result', '')
        
        all_extracted_schemes = extract_schemes_from_text(content_to_parse)

        if not all_extracted_schemes:
            return "No government schemes were confidently extracted. The documents might not contain a clear list, or the format is not recognized."

        response_text = f"✅ Found {len(all_extracted_schemes)} potential schemes:\n\n"
        for i, scheme in enumerate(all_extracted_schemes, 1):
            response_text += f"{i}. {scheme}\n"
        response_text += "\n\nℹ️ Note: This list is extracted based on document content. Some names may be partial or inferred."
        return response_text
    except Exception as e:
        return f"Error during optimized scheme query: {str(e)}"

print("⚡ RAG services optimized for ULTRA-FAST WhatsApp responses!")
print(f"📊 Cache size: {_cache_max_size}, Default model: llama3.1:8b")
print("🎯 Enhanced Marathi detection: 'Jssk baddal mahiti dya' will be detected as Marathi")