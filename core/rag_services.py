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

def get_whatsapp_prompt_template():
    """
    UPDATED prompt template with ENHANCED Devanagari language handling
    """
    template = """You are a female efficient Knowledge Assistant, designed for answering questions specifically from the knowledge base provided to you.

Your task is as follows: give a detailed response for the user query in the user language (e.g., "what are some schemes?" --> "Here is a list of some schemes").

**CRITICAL DEVANAGARI LANGUAGE MATCHING RULES:**
* **IDENTIFY THE EXACT LANGUAGE FIRST**: Carefully analyze if Devanagari text is Marathi or Hindi
* **MARATHI INDICATORS**: आहे, तुम्ही, मी, माहिती, कसा, कसे, बद्दल, द्या, सांगा, मला, पाहिजे, येथे, करा
* **HINDI INDICATORS**: है, आप, मैं, जानकारी, कैसे, कैसा, के बारे में, बताएं, मुझे, चाहिए, यहाँ, करना
* **STRICTLY answer in the EXACT SAME LANGUAGE as the question**
* If question is in English → Answer ONLY in English
* If question is in Hindi (Devanagari) → Answer ONLY in Hindi (Devanagari)
* If question is in Marathi (Devanagari) → Answer ONLY in Marathi (Devanagari)
* **NEVER mix languages or use wrong script - this is mandatory**
* Read numbers as digits, e.g., "104" instead of "one hundred four"
* If the source documents are in different language than question, **translate the information to match question language and script**
* Use direct, everyday language appropriate to the detected language
* Maintain a personal and friendly tone in the user's language
* Provide detailed and comprehensive responses with minimum 150-200 words per scheme
* Include **toll-free numbers** and **complete visible website URLs** *only if those URLs are present in the knowledge base*
* When providing contact information, always include specific contact details from the knowledge base if available

**SCRIPT-SPECIFIC FORMATTING REQUIREMENTS:**
* **Always format your answer using appropriate script and markdown**
* **For Devanagari languages, use proper Devanagari script throughout**
* Use clear section headers based on detected language and script:
  - **English**: "Description", "Eligibility", "Benefits", "How to Apply", "Required Documents", "Contact Information"
  - **Hindi (Devanagari)**: "विवरण", "पात्रता", "लाभ", "आवेदन कैसे करें", "आवश्यक दस्तावेज", "संपर्क जानकारी"
  - **Marathi (Devanagari)**: "वर्णन", "पात्रता", "फायदे", "अर्ज कसा करावा", "आवश्यक कागदपत्रे", "संपर्क माहिती"

**MANDATORY SCRIPT-CORRECT ENDINGS:**
* **For English questions**: "**For more details, please contact the 104/102 helpline numbers.**"
* **For Hindi questions (Devanagari)**: "**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**"
* **For Marathi questions (Devanagari)**: "**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"

**NO RELEVANT CONTEXT RESPONSES (SCRIPT-SPECIFIC):**
* **Marathi question (Devanagari)**: "क्षमस्व, मी या विषयावर तुमची मदत करू शकत नाही. अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा."
* **Hindi question (Devanagari)**: "माफ़ कीजिए, मैं इस विषय पर आपकी मदत नहीं कर सकती। अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।"
* **English question**: "I'm sorry, I cannot assist with that topic. For more details, please contact the 104/102 helpline numbers."

**IMPORTANT CONSTRAINTS:**
* Do not provide answers based on assumptions or general knowledge - USE ONLY KNOWLEDGE BASE
* **Response Length Requirements**: Minimum 150 words for single scheme, 300+ words for multiple schemes
* **If user asks for jokes, casual conversation, or non-scheme topics, respond with appropriate "no relevant context" message above**
* **Contact Information Priority**: Always include specific contact details from knowledge base when available

**DEVANAGARI RESPONSE VALIDATION:**
Before providing your answer, verify:
1. Is the input Marathi or Hindi? (Check for language-specific words)
2. Is the response in the SAME language AND script as the question?
3. Does it use proper Devanagari script for Hindi/Marathi responses?
4. Does it contain information from the knowledge base only?
5. Does it have the correct helpline ending for the language?
6. Is it properly formatted with appropriate script-based headers?

**EXAMPLES OF CORRECT LANGUAGE MATCHING:**
- Input: "मला माहिती पाहिजे" (Marathi) → Output must be in Marathi Devanagari
- Input: "मुझे जानकारी चाहिए" (Hindi) → Output must be in Hindi Devanagari  
- Input: "I need information" (English) → Output must be in English

Here is the content you will work with: {context}

Question: {question}

**CRITICAL: Analyze the question language carefully. If it contains Devanagari script, determine if it's Marathi or Hindi before responding. Match your response language and script EXACTLY to the input.**

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
    
    # ENHANCED DETAILED PROMPT
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

# ENHANCED VALIDATION AND FORMATTING FUNCTIONS
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
            'hi': "नमस्ते! मैं आज आपकी किस प्रकार से मदद कर सकती हूँ? 😊",
            'mr': "नमस्कार! मी तुम्हाला आज कशा प्रकारे मदत करू शकते? 😊",
            'en': "How can I help you today? 😊"
        }
        return greetings_responses.get(normalized_lang, greetings_responses['en'])
    
    return None

# ENHANCED ERROR HANDLING FUNCTIONS
def get_language_specific_error(error_type: str, language: str) -> str:
    """Get language-specific error messages"""
    error_templates = {
        'no_context': {
            'hi': "जानकारी नहीं मिली। कृपया दूसरे तरीके से पूछें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "माहिती मिळाली नाही. कृपया वेगळ्या पद्धतीने विचारा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "No information found. Please rephrase your question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        },
        'technical_error': {
            'hi': "तकनीकी समस्या हुई। पुनः प्रयास करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "तांत्रिक समस्या झाली. पुन्हा प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "Technical issue occurred. Please try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        },
        'timeout_error': {
            'hi': "जवाब धीमा है। छोटा प्रश्न पूछें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "उत्तर धीमे आहे. लहान प्रश्न विचारा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "Response slow. Ask shorter question.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        },
        'system_starting': {
            'hi': "सिस्टम शुरू हो रहा है। कृपया 30 सेकंड में कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "सिस्टम सुरू होत आहे. कृपया 30 सेकंदांनी प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "System initializing. Please try again in 30 seconds.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        },
        'language_mismatch': {
            'hi': "भाषा की समस्या हुई। कृपया फिर से कोशिश करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "भाषेची समस्या झाली. कृपया पुन्हा प्रयत्न करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**",
            'en': "Language processing issue. Please try again.\n\n**For more details, please contact the 104/102 helpline numbers.**"
        }
    }
    
    return error_templates.get(error_type, {}).get(language, error_templates.get(error_type, {}).get('en', 'Error occurred. Please try again.'))

# ADDITIONAL UTILITY FUNCTIONS
def clean_response_text(text: str) -> str:
    """Clean and format response text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert markdown to WhatsApp formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    
    # Ensure proper line breaks
    text = text.replace('\n\n\n', '\n\n')
    
    return text

def is_greeting_query(query: str) -> bool:
    """Check if query is a greeting"""
    greetings = ['hi', 'hello', 'hey', 'namaste', 'नमस्ते', 'नमस्कार', 'हैलो', 'हाय']
    query_lower = query.strip().lower()
    return query_lower in greetings or any(greet in query_lower for greet in greetings)

def is_casual_query(query: str) -> bool:
    """Check if query is casual conversation"""
    casual_patterns = ['joke', 'funny', 'chat', 'talk', 'weather', 'how are you', 'what\'s up']
    query_lower = query.strip().lower()
    return any(pattern in query_lower for pattern in casual_patterns)

# PERFORMANCE MONITORING
def log_performance_metrics(operation: str, duration: float, language: str = None, success: bool = True):
    """Log performance metrics for monitoring"""
    status = "SUCCESS" if success else "FAILED"
    lang_info = f" [{language}]" if language else ""
    print(f"📊 {operation}{lang_info}: {duration:.2f}s - {status}")

# FINAL INITIALIZATION MESSAGE
print("⚡ RAG services optimized for ULTRA-FAST WhatsApp responses!")
print(f"📊 Cache size: {_cache_max_size}, Default model: llama3.1:8b")
print("🔍 Enhanced Devanagari language detection enabled")
print("✅ All language validation functions loaded")
print("🚀 Ready for production WhatsApp deployment!")