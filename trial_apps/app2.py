import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import tempfile

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
    index=3  # Marathi default
)
language_code = lang_map[language]


# Audio or file input
audio_value = st.audio_input("Speak your query:")
uploaded_file = st.file_uploader("Or upload a query audio file:", type=["wav", "mp3", "m4a"])

file_source = audio_value if audio_value else uploaded_file

if file_source:
    with st.spinner("Processing your query..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(file_source.getvalue())
            file_path = temp_audio_file.name

        with open(file_path, "rb") as file:
            transcription_args = dict(
                file=file,
                model="whisper-large-v3",
                temperature=0.0,
                response_format="verbose_json"
            )
            if language_code:
                transcription_args["language"] = language_code

            transcription = client.audio.transcriptions.create(**transcription_args)

        query_text = transcription.text
        st.subheader("Transcribed Query:")
        st.write(query_text)
        
    os.remove(file_path)
else:
    st.write("Please provide an audio input or upload a file to proceed.")
