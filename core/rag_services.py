import tempfile
import os
import time
import hashlib
import re
import langid
from typing import Optional, List, Dict, Any
import google.generativeai as genai
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
    DetectorFactory.seed = 0  # for consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not installed. Using langid only for language detection.")
    LANGDETECT_AVAILABLE = False

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "mr": "Marathi"
}

set_verbose(False)

_query_cache = {}
_cache_max_size = 50

class GeminiLLM(LLM):
    """Custom LangChain LLM wrapper for Google Gemini models"""
    
    model: Any = None
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 1500
    temperature: float = 0.05
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model with safety settings
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    @property
    def _llm_type(self) -> str:
        return "google_gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Google Gemini model"""
        try:
            response = self.model.generate_content(prompt)
            
            # Handle the response
            if response.text:
                return response.text.strip()
            else:
                # Handle safety filter blocks
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return "I apologize, but I cannot provide a response to that query due to safety guidelines. Please rephrase your question."
                    elif reason == "MAX_TOKENS":
                        return "Response was truncated due to length limits. Please ask a more specific question."
                
                return "I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            print(f"❌ Gemini call failed: {e}")
            return f"Error: Unable to process request. Please try again later."

def get_query_hash(query_text):
    """Generate a hash for caching queries"""
    return hashlib.md5(query_text.encode()).hexdigest()

def cache_result(query_hash, result):
    """Cache query result"""
    global _query_cache
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest entry (FIFO)
        try:
            oldest_key = next(iter(_query_cache))
            del _query_cache[oldest_key]
        except StopIteration:
            pass
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available"""
    return _query_cache.get(query_hash)

def clear_query_cache():
    """Clear the RAG query cache"""
    global _query_cache
    _query_cache.clear()
    print("RAG query cache cleared.")

def detect_language_langid(text):
    """Return the ISO 639-1 language code and full language name, if supported."""
    lang, _ = langid.classify(text)
    return lang, SUPPORTED_LANGUAGES.get(lang, "Unsupported")

def detect_language(text):
    """
    Auto-detect language from text
    Returns language code (e.g., 'en', 'hi', 'mr')
    Uses langdetect as primary, langid as fallback
    """
    try:
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < 3:
            return 'en'  # Default for very short text
        
        detected_lang = 'en'  # Default
        
        # Try langdetect first (more accurate for these languages)
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(clean_text)
            except:
                # Fallback to langid
                detected_lang, _ = langid.classify(clean_text)
        else:
            # Use langid as primary
            detected_lang, _ = langid.classify(clean_text)
        
        # Map to supported languages
        lang_mapping = {
            'hi': 'hi',
            'mr': 'mr', 
            'en': 'en'
        }
        
        return lang_mapping.get(detected_lang, 'en')
        
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'  # Safe default

def get_whatsapp_prompt_template():
    """
    Single source of truth for the WhatsApp-formatted prompt template
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
* Use section headers like:
  - **English**: "Description", "Eligibility", "Benefits", "How to Apply", "Required Documents", "Contact Information"
  - **Hindi**: "विवरण", "पात्रता", "लाभ", "आवेदन कैसे करें", "आवश्यक दस्तावेज", "संपर्क जानकारी"
  - **Marathi**: "वर्णन", "पात्रता", "फायदे", "अर्ज कसा करावा", "आवश्यक कागदपत्रे", "संपर्क माहिती"

* **Always format your answer using markdown. Use markdown headings (##), bold (**text**), bullet lists (-), and other markdown features where appropriate. This applies to English, Hindi, and Marathi answers.**

* **MANDATORY ENDING**: Always end every response with the helpline information in the user's language:
  - **In English**: "**For more details, please contact the 104/102 helpline numbers.**"
  - **In Hindi**: "**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**"
  - **In Marathi**: "**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"

* If there is no relevant context for the question, simply say:
  - **In Marathi**: "क्षमस्व, मी या विषयावर तुमची मदत करू शकत नाही. अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा."
  - **In Hindi**: "माफ़ कीजिए, मैं इस विषय पर आपकी मदत नहीं कर सकती। अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।"
  - **In English**: "I'm sorry, I cannot assist with that topic. For more details, please contact the 104/102 helpline numbers."

* **STRICT LANGUAGE MATCHING**: 
  - Hindi input → Hindi output ONLY
  - English input → English output ONLY
  - Marathi input → Marathi output ONLY
  - Never mix languages or use wrong language

* **Remove duplicate information and provide only one consolidated answer.**
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

def get_safe_model_config(model_choice="gemini-1.5-flash"):
    """Get safe configuration optimized for Google Gemini"""
    return {
        "max_tokens": 1500,  # Safe limit for Gemini
        "chunk_size": 400,   # Smaller chunks for better context management
        "max_chunks": 8      # Limited chunks to prevent token overflow
    }

def build_rag_chain_from_files(pdf_file, txt_file, google_api_key, model_name="gemini-1.5-flash", 
                              use_local=False, enhanced_mode=True):
    """
    Build a RAG chain from PDF and/or TXT files using Google Gemini
    """
    print(f"🤖 Using Google Gemini model: {model_name}")
    
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    temp_files_to_clean = []
    try:
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.getvalue())
                pdf_path = tmp_pdf.name
                temp_files_to_clean.append(pdf_path)
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.getvalue())
                txt_path = tmp_txt.name
                temp_files_to_clean.append(txt_path)

        all_docs = []
        if pdf_path:
            all_docs.extend(PyPDFLoader(pdf_path).load())
        if txt_path:
            all_docs.extend(TextLoader(txt_path, encoding="utf-8").load())
        
        if not all_docs:
            raise ValueError("No valid documents loaded or documents are empty.")

        # Get optimized configuration
        config = get_safe_model_config()
        print(f"📊 Using config: {config}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=max(50, int(config["chunk_size"] * 0.15)),
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )
        splits = splitter.split_documents(all_docs)
        
        if not splits:
            raise ValueError("Document splitting resulted in no chunks.")
            
        retriever = TFIDFRetriever.from_documents(splits, k=min(config["max_chunks"], len(splits)))

        custom_prompt = get_whatsapp_prompt_template()
        
        chain_kwargs = {
            "prompt": custom_prompt,
            "verbose": False,
        }
        
        # Create Google Gemini LLM
        llm = GeminiLLM(
            api_key=google_api_key,
            model_name=model_name,
            max_tokens=config["max_tokens"],
            temperature=0.05,
            callbacks=[StrictContextCallback()]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs=chain_kwargs
        )
            
    finally:
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.unlink(f_path)

def build_rag_chain_from_documents(documents, google_api_key, model_name="gemini-1.5-flash", 
                                 use_local=False, enhanced_mode=True):
    """
    Build RAG chain from in-memory documents using Google Gemini
    """
    print(f"🔧 Building RAG chain with Google Gemini model: {model_name}")
    
    # Convert each document to a LangChain Document object
    all_docs = [
        Document(
            page_content=doc["content"], 
            metadata={"filename": doc.get("filename", "unknown")}
        ) 
        for doc in documents
    ]
    
    if not all_docs:
        raise ValueError("No documents loaded in memory.")
    
    # Get optimized configuration
    config = get_safe_model_config()
    print(f"📊 Using config: {config}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=max(50, int(config["chunk_size"] * 0.15)),
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len
    )
    splits = splitter.split_documents(all_docs)
    print(f"📄 Created {len(splits)} document chunks")
    
    if not splits:
        raise ValueError("Document splitting resulted in no chunks.")
    
    retriever = TFIDFRetriever.from_documents(splits, k=min(config["max_chunks"], len(splits)))
    
    custom_prompt = get_whatsapp_prompt_template()
    
    chain_kwargs = {
        "prompt": custom_prompt,
        "verbose": False
    }
    
    # Create Google Gemini LLM
    try:
        llm = GeminiLLM(
            api_key=google_api_key,
            model_name=model_name,
            max_tokens=config["max_tokens"],
            temperature=0.05,
            callbacks=[StrictContextCallback()]
        )
        print(f"✅ Google Gemini LLM created successfully")
        
    except Exception as e:
        print(f"❌ Error creating Google Gemini LLM: {e}")
        raise
    
    try:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs=chain_kwargs
        )
        print("✅ RAG chain created successfully with Google Gemini")
        return rag_chain
        
    except Exception as e:
        print(f"❌ Error creating RAG chain: {e}")
        raise

def process_scheme_query_with_retry(rag_chain, user_query, max_retries=2, enable_tts=False, autoplay=False):
    """
    Process query with rate limit handling, caching, and better error handling
    Returns: (text_result, audio_html, language_detected, cache_info)
    """
    # Detect language and enforce allowed languages
    supported_languages = {"en", "hi", "mr"}
    detected_lang = detect_language(user_query)
    if detected_lang not in supported_languages:
        return (
            "⚠️ Sorry, only Marathi, Hindi, and English are supported. कृपया मराठी, हिंदी अथवा इंग्रजी भाषेत विचारा.",
            "",
            detected_lang,
            {"text_cache": "skipped", "audio_cache": "not_generated"}
        )

    # Check cache first
    query_hash = get_query_hash(user_query.lower().strip())
    cached_result = get_cached_result(query_hash)
    if cached_result:
        result_text = cached_result
        cache_status = "cached"
    else:
        cache_status = "not_cached"
        
        # Limit query length to prevent token issues
        original_query = user_query
        if len(user_query) > 200:
            user_query = user_query[:200] + "..."
            print(f"⚠️ Query truncated from {len(original_query)} to {len(user_query)} chars")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1} of {max_retries}")
                
                result = rag_chain.invoke({"query": user_query})
                if isinstance(result, dict):
                    result_text = result.get('result', 'No results found.')
                elif isinstance(result, tuple) and len(result) > 0:
                    result_text = str(result[0])
                else:
                    result_text = str(result)
                
                # Cache the result
                cache_result(query_hash, result_text)
                cache_status = "cached"
                break
                
            except Exception as e:
                error_str = str(e).lower()
                print(f"❌ Attempt {attempt + 1} failed: {error_str}")
                
                if "quota" in error_str or "limit" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"⏱️ Rate limited, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        result_text = "Daily limit reached. Please try again tomorrow. For more details, please contact the 104/102 helpline numbers."
                        break
                
                elif "safety" in error_str:
                    result_text = "I apologize, but I cannot provide a response to that query. Please rephrase your question. For more details, contact 104/102 helpline numbers."
                    break
                
                elif "token" in error_str or "too large" in error_str:
                    if attempt < max_retries - 1:
                        user_query = user_query[:100] + "..." if len(user_query) > 100 else user_query
                        print(f"🔄 Retrying with shorter query: {len(user_query)} chars")
                        continue
                    else:
                        result_text = "Unable to answer right now, please try again after sometime. For more details, please contact the 104/102 helpline numbers."
                        break
                
                else:
                    if attempt < max_retries - 1:
                        print(f"🔄 Unknown error, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        result_text = "Unable to answer right now, please try again after sometime. For more details, please contact the 104/102 helpline numbers."
                        break
        else:
            result_text = "Unable to process query after multiple attempts. Please try a simpler question."
    
    return (result_text, "", detected_lang, {"text_cache": cache_status, "audio_cache": "not_generated"})

def get_model_options():
    """Return available Google Gemini model options"""
    return {
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash (Recommended)", 
            "description": "Fast, high-quality model with generous free tier - 1500 requests/day"
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro", 
            "description": "Most capable model for complex queries - 50 requests/day free"
        },
        "gemini-pro": {
            "name": "Gemini Pro", 
            "description": "Previous generation model - reliable performance"
        }
    }

class StrictContextCallback(BaseCallbackHandler):
    """Callback to monitor and log RAG responses"""
    
    def on_chain_end(self, outputs, **kwargs):
        """Check if response might contain external knowledge"""
        response = outputs.get("result", "")
        suspicious_phrases = [
            "I believe",
            "generally",
            "typically", 
            "in most cases",
            "commonly",
            "as per my knowledge",
            "based on my understanding"
        ]
        
        for phrase in suspicious_phrases:
            if phrase.lower() in response.lower():
                print(f"Warning: Response may contain external knowledge. Suspicious phrase: {phrase}")

# Backward compatibility functions
def build_rag_chain_with_model_choice(pdf_file, txt_file, google_api_key, model_choice="gemini-1.5-flash", 
                                     use_local=False, enhanced_mode=True):
    """
    Build RAG chain with model choice - backward compatibility function
    """
    return build_rag_chain_from_files(pdf_file, txt_file, google_api_key, model_choice, use_local, enhanced_mode)

# Extract schemes functionality (kept for compatibility)
def extract_schemes_from_text(text_content):
    """Helper function to extract schemes from text content using regex patterns."""
    # Normalize text: collapse multiple whitespaces, handle newlines
    text = re.sub(r'\s+', ' ', str(text_content)).replace('\n', ' ')

    # More comprehensive patterns, including common Marathi and English scheme indicators
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
        # A broad query to fetch relevant context containing scheme lists
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