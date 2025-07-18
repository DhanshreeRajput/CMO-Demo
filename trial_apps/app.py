import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import tempfile
import json
from difflib import get_close_matches

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

from difflib import get_close_matches

def find_relevant_schemes(query, kb, top_k=3):
    titles = [entry["title"] for entry in kb]
    matches = get_close_matches(query, titles, n=top_k, cutoff=0.3)
    return [entry for entry in kb if entry["title"] in matches]

def format_scheme(entry):
    eligibility = entry.get("eligibility", {})
    return (
        f"Title: {entry.get('title', '')}\n"
        f"Description: {entry.get('description', '')}\n\n"
        f"Eligibility:\n"
        f"  Target Group: {eligibility.get('target_group', '')}\n"
        f"  Inclusion Criteria: {eligibility.get('inclusion_criteria', '')}\n"
        f"  Exclusion Criteria: {eligibility.get('exclusion_criteria', '')}\n"
        f"  Special Exclusion Criteria: {eligibility.get('special_exclusion_criteria', '')}\n\n"
        f"Benefits:\n{entry.get('benefits', '')}"
    )

def search_knowledge_base(query):
    """Search through knowledge base using fuzzy title matching"""
    try:
        relevant_entries = find_relevant_schemes(query, knowledge_base)
        if not relevant_entries:
            return "No relevant schemes found for the query."
        return "\n\n---\n\n".join([format_scheme(entry) for entry in relevant_entries])
    except Exception as e:
        return f"Error in knowledge base search: {str(e)}"

    """Search through knowledge base using fuzzy title matching"""
    try:
        relevant_entries = find_relevant_schemes(query, knowledge_base)
        if not relevant_entries:
            return "No relevant schemes found for the query."

        context_snippets = [
            f"Title: {entry['title']}\nDescription: {entry['description']}\nBenefits: {entry['benefits']}"
            for entry in relevant_entries
        ]
        return "\n\n".join(context_snippets)
    except Exception as e:
        return f"Error in knowledge base search: {str(e)}"

def generate_answer(context, query):
    """Generate final answer using Groq's LLM"""
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{
                "role": "system",
                "content": f"You're a government help assistant. Use this context: {context}"
            },
            {
                "role": "user",
                "content": query
            }],
            temperature=0.0,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

audio_value = st.audio_input("What is your query?")

if audio_value:
    with st.spinner("Processing your query..."):
        # Save and transcribe audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_value.getvalue())
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

        # Knowledge base search
        context = search_knowledge_base(query_text)

        # Generate final answer
        answer = generate_answer(context, query_text)

        st.subheader("AI Response:")
        st.write(answer)

        os.remove(file_path)
else:
    st.write("Please provide an audio input to proceed.")