import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain import build_rag_chain_from_files
import tempfile

load_dotenv()

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def transcribe_audio(client, audio_bytes):
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
        return transcription.text
    finally:
        os.unlink(temp_path)

def main():
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("🤖 RAG Assistant – English & Marathi Support")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please set it in your .env file.")
        st.stop()

    init_session_state()

    # Upload files
    st.markdown("### 📄 Upload Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload English PDF", type=["pdf"])
    uploaded_txt = st.file_uploader("Upload Marathi Text File", type=["txt"])

    if not (uploaded_pdf and uploaded_txt):
        st.warning("Please upload both PDF and TXT files to continue.")
        st.stop()

    # Load Whisper Client and RAG
    whisper_client = Groq(api_key=GROQ_API_KEY)
    rag_chain = build_rag_chain_from_files(uploaded_pdf, uploaded_txt, GROQ_API_KEY)

    # Input Section
    st.markdown("### Ask a question by typing or using audio input")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Enter your question", key="text_input", placeholder="e.g. योजना माहिती द्या...")
    with col2:
        audio_value = st.audio_input("🎤 Record your query")

    user_text = None
    if audio_value is not None:
        try:
            user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    if st.button("🔍 Get Answer") or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            try:
                assistant_reply = rag_chain.invoke(input_text)["result"]
                st.session_state.chat_history.insert(0, {"user": input_text, "assistant": assistant_reply})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Chat history
    with st.expander("📜 Chat History", expanded=True):
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.markdown(
                    f"""<div style='background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:5px;'>
                    <strong>🧑 You:</strong> {entry['user']}
                    </div>""", unsafe_allow_html=True
                )
                st.markdown(
                    f"""<div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:15px;'>
                    <strong>🤖 Assistant:</strong> {entry['assistant']}
                    </div>""", unsafe_allow_html=True
                )
        else:
            st.info("No chat history yet. Ask your first question!")

if __name__ == "__main__":
    main()
