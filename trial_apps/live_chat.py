import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from groq import Groq
import json

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def chat_with_llama(client, prompt, history):
    messages = [{"role": "system", "content": "You are a Maharashtra helpline assistant.Users call you in Marathi,hindi and English languages. Answer in the same language as the user input. example: hey!"}]
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Chat failed: {e}")
        return None

def transcribe_audio(client, audio_bytes):
    # Save temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                temperature=0.0
            )
        os.unlink(temp_path)  # Clean up temp file
        return transcription.text
    except Exception as e:
        os.unlink(temp_path)  # Clean up temp file even if error occurs
        raise e

def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Please set it in your environment variables.")
        st.stop()

    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)

    # App UI
    st.set_page_config(page_title="🎤💬 Live Chat", layout="wide")
    st.title("🎤💬 CMO Live Chat (Audio & Text)")

    # Initialize session state
    init_session_state()

    # User input interface
    st.markdown("#### Type your message or record audio:")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Text Input", key="text_input", placeholder="Type your message...")
    with col2:
        audio_value = st.audio_input("Record your query")

    # Process audio input directly if available
    if audio_value is not None:
        try:
            user_text = transcribe_audio(client, audio_value.getvalue())
            st.success(f"Transcribed Audio: {user_text}")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

        if user_text:
            assistant_reply = chat_with_llama(client, user_text, st.session_state.chat_history)
            if assistant_reply:
                st.session_state.chat_history.insert(0, {
                    "user": user_text,
                    "assistant": assistant_reply
                })

    submit = st.button("Send")

    # Handle submission
    if submit and user_input.strip():
        user_text = user_input.strip()
        assistant_reply = chat_with_llama(client, user_text, st.session_state.chat_history)
        if assistant_reply:
            st.session_state.chat_history.insert(0, {
                "user": user_text,
                "assistant": assistant_reply
            })

    # Display chat history in a chat-like format
    with st.empty().container():
        for entry in (st.session_state.chat_history):
            st.markdown(f'<div style="padding: 5px; border-radius: 5px; margin-bottom: 5px; background-color: #E0F2F7;"><strong>🤓 You:</strong> {entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="padding: 5px; border-radius: 5px; margin-bottom: 5px; background-color: #F0F0F0;"><strong>🤖 Assistant:</strong> {entry["assistant"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()