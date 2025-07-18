import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import tempfile
import json
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("CMO AI")
st.write("Supported languages: Marathi, Hindi, English")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Language selection with Auto Detect as default
lang_map = {
    "Auto Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}
language = st.selectbox(
    "Select your query language:",
    list(lang_map.keys()),
    index=0  # Auto Detect is default
)
language_code = lang_map[language]
# Load knowledge base as python list
with open("knowledge.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

def search_knowledge_base(query):
    """Search through knowledge base using Groq's NLP capabilities"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": f"Search this knowledge base: {json.dumps(knowledge_base)}. Return only relevant information for: {query}"
            }],
            temperature=0.3,
            max_tokens=528
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in knowledge base search: {str(e)}"

def generate_answer(context, query):
    """Generate final answer using Groq's LLM"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": f"You're a government help assistant. Use this context: {context}"
            },
            {
                "role": "user",
                "content": query
            }],
            temperature=0.5,
            max_tokens=1024
            
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Load Whisper model and processor once
MODEL_PATH = r"D:\Dhanshree and Team A.I\Gitlab\CMO-AI\model\hf_models\whisper_large_v3_model"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

audio_value = st.audio_input("What is your query?")

if audio_value:
    with st.spinner("Processing your query..."):
        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_value.getvalue())
            file_path = temp_audio_file.name

        # Load and preprocess audio
        speech_array, sampling_rate = torchaudio.load(file_path)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
        input_features = processor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Set language if not auto-detect
        forced_decoder_ids = None
        if language_code:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_code, task="transcribe")

        # Transcribe
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            query_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        st.subheader("Transcribed Query:")
        st.write(query_text)

        # Knowledge base search
        context = search_knowledge_base(query_text)
        
        # Generate final answer
        answer = generate_answer(context, query_text)
        
        st.subheader("AI Response:")
        st.write(answer)

        os.remove(file_path)
else:
    st.write("Please provide an audio input to proceed.")
