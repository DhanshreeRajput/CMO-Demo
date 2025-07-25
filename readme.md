# CMRF AI WhatsApp Assistant

A multilingual WhatsApp AI assistant built with FastAPI and Groq AI that provides information about government schemes and services. Supports English, Hindi, and Marathi languages.

## Features

- 🌐 **Multilingual Support**: Handles queries in English, Hindi, and Marathi
- 📚 **Document Processing**: Supports PDF and TXT file uploads for knowledge base
- 🤖 **AI-Powered Responses**: Uses Groq AI for intelligent responses
- ⚡ **Rate Limiting**: Built-in rate limiting (5 seconds) to prevent spam
- 💾 **Redis Caching**: Efficient caching and rate limiting with Redis
- 🔄 **Message Deduplication**: Prevents duplicate message processing
- 📊 **Comprehensive Monitoring**: Built-in health checks and usage statistics

## Prerequisites

- Python 3.9+
- Redis Server
- Groq API Key
- WhatsApp Business API credentials
- ngrok (for local development)

## Environment Variables

Create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_groq_api_key
WHATSAPP_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_verify_token
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=optional_redis_password
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CMO-Whatsapp.git
cd CMO-Whatsapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Redis server:
```bash
# On Windows
redis-server

# On Linux/Mac
sudo service redis start
```

4. Start the FastAPI application:
```bash
uvicorn fastapp:app --host 0.0.0.0 --port 8080
```

## Project Structure

```
CMO-Whatsapp/
├── core/
│   ├── __init__.py
│   ├── rag_services.py      # RAG chain and language detection
│   └── transcription.py     # Language processing
├── utils/
│   ├── __init__.py
│   └── config.py           # Configuration management
├── uploads/                # Document storage
├── fastapp.py             # Main FastAPI application
├── requirements.txt       # Project dependencies
└── README.md
```

## API Endpoints

### Core Endpoints
- `GET /`: Root endpoint with system status
- `POST /upload/`: Upload PDF/TXT files to knowledge base
- `POST /query/`: Query the knowledge base
- `GET /webhook`: WhatsApp webhook verification
- `POST /webhook`: WhatsApp message handling

### Status Endpoints
- `GET /health/`: Basic health check
- `GET /api/status/comprehensive`: Detailed system status
- `GET /api/status/quick`: Quick status check
- `GET /usage/stats`: Usage statistics

### WhatsApp Endpoints
- `GET /whatsapp/health`: WhatsApp integration status
- `POST /whatsapp/test`: Test WhatsApp messaging

## Features in Detail

### Document Processing
- Supports PDF and TXT file uploads
- Automatic document chunking and vectorization
- TF-IDF based retrieval for relevant context

### Language Support
- Automatic language detection
- Multi-language response generation
- Support for Devanagari script

### Rate Limiting
- 5-second cooldown between messages
- Redis-based rate limiting
- Prevents message spam and loops

### Monitoring
- Comprehensive health checks
- Usage statistics tracking
- System resource monitoring
- Service availability checks

## Security Features

- CORS middleware configured
- Rate limiting implemented
- Input validation
- Error handling
- Message deduplication
- Environment variable validation

## Development

### Running Tests
```bash
# TODO: Add test commands
```

### Local Development
1. Start ngrok:
```bash
ngrok http 8080
```

2. Update WhatsApp webhook URL with ngrok URL
3. Start the application in debug mode:
```bash
uvicorn fastapp:app --reload
```

## Docker Support

Build and run with Docker:

```bash
docker build -t cmrf-whatsapp -f Dockerfile.fastapiwhatsapp .
docker run -p 8080:8080 --env-file .env cmrf-whatsapp
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Verify Redis is running
   - Check Redis connection settings

2. **WhatsApp Webhook Issues**
   - Verify webhook URL is accessible
   - Check WhatsApp credentials

3. **Knowledge Base Empty**
   - Upload documents via /upload/ endpoint
   - Check file format support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the web framework
- Groq AI for language processing
- WhatsApp Business API for messaging
- Redis for caching and rate limiting
