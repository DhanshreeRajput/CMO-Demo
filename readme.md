# SAMNEX WhatsApp AI Assistant - Google Gemini Edition

A multilingual WhatsApp AI assistant built with FastAPI and Google Gemini AI that provides information about government schemes and services. Supports English, Hindi, and Marathi languages with advanced RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 🌐 **Multilingual Support**: Handles queries in English, Hindi, and Marathi with automatic language detection
- 📚 **Document Processing**: Supports PDF and TXT file uploads for knowledge base creation
- 🤖 **Google Gemini AI**: Powered by Google's Gemini 1.5 Flash model with FREE tier (1500 requests/day)
- 🎤 **Voice Support**: Audio transcription using Whisper for voice messages
- 📝 **RAG Implementation**: Advanced retrieval-augmented generation with TF-IDF vectorization
- ⚡ **Rate Limiting**: Built-in rate limiting (5 seconds) to prevent spam and manage API quotas
- 💾 **Redis Caching**: Efficient query caching and rate limiting with Redis
- 🔄 **Message Deduplication**: Prevents duplicate message processing
- 📊 **Comprehensive Monitoring**: Built-in health checks, usage statistics, and API status monitoring
- 🎯 **Smart Response Formatting**: WhatsApp-optimized markdown formatting with proper structure

## 🤖 AI Model Information

### Google Gemini 1.5 Flash
- **Provider**: Google AI
- **Model**: `gemini-1.5-flash` (recommended)
- **Cost**: **100% FREE**
- **Daily Limit**: 1,500 requests per day
- **Rate Limit**: 15 requests per minute
- **Context Window**: Large context support for document processing
- **Languages**: Excellent support for Hindi, Marathi, and English
- **Safety**: Built-in safety filters and content moderation

### Alternative Models Available
- **Gemini 1.5 Pro**: More capable for complex queries (50 requests/day free)
- **Gemini Pro**: Previous generation model (reliable performance)

### Cost Comparison
| Provider | Model | Daily Limit | Monthly Cost | Notes |
|----------|-------|-------------|--------------|-------|
| Google Gemini | 1.5 Flash | 1,500 requests | **FREE** | Recommended |
| Google Gemini | 1.5 Pro | 50 requests | **FREE** | Complex queries |
| OpenAI | GPT-3.5 | ~3,000 requests | $5-20 | Paid only |
| Groq | Llama/Mixtral | 14,400 requests | **FREE** | Alternative |

## 📋 Prerequisites

- Python 3.9+
- Redis Server
- **Google AI API Key** (FREE - get from [Google AI Studio](https://makersuite.google.com/))
- WhatsApp Business API credentials
- ngrok (for local development)

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```env
# Google Gemini AI Configuration
GOOGLE_API_KEY=your_google_ai_api_key
MODEL_NAME=gemini-1.5-flash
AI_PROVIDER=google

# WhatsApp Business API Configuration
WHATSAPP_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_verify_token

# Redis Configuration (Optional - fallback available)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=optional_redis_password
```

### 🆓 Getting Google AI API Key (FREE)

1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" → "Create API Key"
4. Copy the API key and add to your `.env` file
5. **No credit card required** - completely free with generous limits!

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/SAMNEX-WhatsApp-Assistant.git
cd SAMNEX-WhatsApp-Assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start Redis server (Optional but recommended):**
```bash
# On Windows with Redis installed
redis-server

# On Linux/Mac
sudo service redis start

# Using Docker
docker run -d -p 6379:6379 redis:alpine
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Start the FastAPI application:**
```bash
python fastapp.py
# or
uvicorn fastapp:app --host 0.0.0.0 --port 8080 --reload
```

## 📁 Project Structure

```
SAMNEX-WhatsApp-Assistant/
├── core/
│   ├── __init__.py
│   ├── rag_services.py         # Google Gemini RAG implementation
│   └── transcription.py        # Audio processing & language detection
├── uploads/                    # Document storage directory
├── fastapp.py                  # Main FastAPI application
├── requirements.txt            # Project dependencies
├── .env.example               # Environment variables template
├── Dockerfile.fastapiwhatsapp # Docker configuration
├── whatsapp.log              # Application logs
└── README.md                 # This file
```

## 🌐 API Endpoints

### 📱 Core Application Endpoints
- `GET /`: Root endpoint with system status and configuration
- `POST /upload/`: Upload PDF/TXT files to build knowledge base
- `POST /query/`: Query the knowledge base with text input
- `POST /voice-query/`: Process voice messages with transcription
- `GET /health/`: Basic health check and service status

### 📞 WhatsApp Integration Endpoints
- `GET /webhook`: WhatsApp webhook verification
- `POST /webhook`: Handle incoming WhatsApp messages
- `POST /whatsapp/test`: Send test WhatsApp messages

### 📊 Monitoring & Debug Endpoints
- `GET /api/status/gemini`: Google Gemini API status check
- `GET /usage/stats`: Usage statistics and quota monitoring
- `POST /debug/test-query`: Test query processing pipeline
- `POST /debug/test-audio-transcription`: Test voice transcription

### 🔧 Management Endpoints
- `POST /clear-cache`: Clear query cache
- `GET /debug/documents`: View loaded knowledge base documents

## 🎯 Features in Detail

### 📚 Document Processing & RAG
- **Supported Formats**: PDF, TXT files
- **Chunking Strategy**: Recursive character splitting with 400 token chunks
- **Retrieval Method**: TF-IDF vectorization with cosine similarity
- **Context Management**: Smart chunk selection (max 8 chunks) to prevent token overflow
- **Language Preservation**: Maintains original document language while translating responses

### 🗣️ Language Support
- **Automatic Detection**: Uses multiple detection methods (langdetect + langid + script analysis)
- **Supported Languages**: 
  - English (en)
  - Hindi (hi) - हिंदी
  - Marathi (mr) - मराठी
- **Script Support**: Full Devanagari script processing
- **Response Matching**: Responds in the same language as the query

### 🎤 Voice Processing
- **Transcription Engine**: OpenAI Whisper (local processing)
- **Audio Formats**: WAV, MP3, M4A, and other common formats
- **Language Detection**: Post-transcription language validation
- **Privacy**: Local processing - audio never sent to external APIs

### ⚡ Performance & Optimization
- **Query Caching**: LRU cache with 50 query limit
- **Rate Limiting**: 5-second cooldown between user messages
- **Message Deduplication**: Prevents processing duplicate WhatsApp messages
- **Token Management**: Smart context window management for Google Gemini
- **Retry Logic**: Automatic retry with exponential backoff for API failures

### 📝 Response Formatting
- **WhatsApp Optimized**: Markdown formatting converted for WhatsApp
- **Structured Responses**: Consistent sections (Description, Eligibility, Benefits, etc.)
- **Mandatory Elements**: Every response includes 104/102 helpline information
- **Length Requirements**: Minimum 150 words per scheme, 300+ for multiple schemes
- **Contact Integration**: Includes specific contact details from knowledge base

## 🔒 Security Features

- **CORS Configuration**: Proper cross-origin resource sharing setup
- **Input Validation**: Comprehensive request validation with Pydantic
- **Rate Limiting**: Multiple layers (Redis + application level)
- **Error Handling**: Graceful error handling with user-friendly messages
- **Message Filtering**: Blocks off-topic queries and inappropriate content
- **Environment Security**: Secure environment variable handling

## 📊 Monitoring & Analytics

### 📈 Usage Tracking
```python
# Current usage statistics available at /usage/stats
{
    "messages_sent_total": 150,
    "messages_received_total": 200,
    "messages_sent_today": 25,
    "gemini_api_calls": 180,
    "uptime_hours": 12.5,
    "daily_limit_status": "Normal"
}
```

### 🏥 Health Monitoring
- **Google Gemini API**: Real-time status and response time monitoring
- **Redis Connection**: Cache availability and performance
- **Knowledge Base**: Document count and last update tracking
- **Rate Limits**: Current usage vs daily/hourly limits

## 🚢 Docker Support

### Build and run with Docker:

```bash
# Build the image
docker build -t samnex-whatsapp -f Dockerfile.fastapiwhatsapp .

# Run with environment file
docker run -p 8080:8080 --env-file .env samnex-whatsapp

# Run with Redis
docker-compose up -d
```

### Docker Compose Example:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - REDIS_HOST=redis
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🛠️ Development

### 🧪 Local Development Setup

1. **Start ngrok for WhatsApp webhook:**
```bash
ngrok http 8080
```

2. **Update WhatsApp webhook URL:**
   - Go to Meta for Developers
   - Update webhook URL to your ngrok URL + `/webhook`
   - Set verify token from your `.env` file

3. **Run in development mode:**
```bash
uvicorn fastapp:app --reload --log-level debug
```

### 🧪 Testing

#### Test Google Gemini API:
```bash
curl -X POST "http://localhost:8080/debug/test-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What schemes are available?"}'
```

#### Test WhatsApp messaging:
```bash
curl -X POST "http://localhost:8080/whatsapp/test" \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+1234567890", "message": "Test message"}'
```

#### Upload knowledge base:
```bash
curl -X POST "http://localhost:8080/upload/" \
  -F "pdf_file=@document.pdf" \
  -F "txt_file=@schemes.txt"
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. **Google Gemini API Issues**
```python
# Check API status
GET /api/status/gemini

# Common errors:
# - "Invalid API key" → Check GOOGLE_API_KEY in .env
# - "Quota exceeded" → Wait for daily reset or upgrade plan
# - "Safety filter" → Rephrase query to avoid triggering filters
```

#### 2. **WhatsApp Webhook Issues**
```bash
# Verify webhook setup
GET /webhook?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=123

# Common issues:
# - Webhook not reachable → Check ngrok/server status
# - Wrong verify token → Update WHATSAPP_VERIFY_TOKEN
# - Messages not responding → Check logs in whatsapp.log
```

#### 3. **Knowledge Base Issues**
```python
# Check document status
GET /debug/documents

# Solutions:
# - No documents → Upload via /upload/ endpoint
# - Empty responses → Check document content and format
# - Language mismatch → Verify document language matches query
```

#### 4. **Redis Connection Issues**
```python
# Check Redis status in health endpoint
GET /health/

# Solutions:
# - Redis unavailable → Start Redis server or use fallback mode
# - Connection refused → Check REDIS_HOST and REDIS_PORT
# - Auth failed → Verify REDIS_PASSWORD if set
```

### 📝 Logs and Debugging

```bash
# View application logs
tail -f whatsapp.log

# Check Redis logs (if using Docker)
docker logs redis-container

# Monitor API usage
curl -s "http://localhost:8080/usage/stats" | jq
```

## 💰 Cost Analysis & Optimization

### Free Tier Limits (Google Gemini)
- **Daily Requests**: 1,500 per day
- **Rate Limit**: 15 per minute
- **Monthly Cost**: $0 (completely free)
- **Context Window**: Up to 1M tokens

### Optimization Tips
1. **Enable Caching**: Reduces API calls by ~60%
2. **Smart Chunking**: Optimized chunk size (400 tokens) prevents overflow
3. **Rate Limiting**: Prevents quota exhaustion
4. **Fallback Responses**: Graceful degradation when limits reached

### Scaling Considerations
- **Free Tier**: Suitable for 1,500 queries/day (~50 users)
- **Production**: Consider Gemini Pro or paid tiers for higher volume
- **Monitoring**: Set up alerts for quota usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### 📋 Development Guidelines
- Follow PEP 8 for Python code
- Add type hints for new functions
- Update tests for new features
- Document API changes
- Test with all supported languages

## 🙏 Acknowledgments

- **Google AI** for providing free access to Gemini models
- **FastAPI** for the excellent web framework
- **Meta** for WhatsApp Business API
- **OpenAI** for Whisper transcription model
- **LangChain** for RAG implementation tools
- **Redis** for caching and rate limiting



