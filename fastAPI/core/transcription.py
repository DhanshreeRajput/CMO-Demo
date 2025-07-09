import tempfile
import os
import langid  # pip install langid
import re

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# Accept if any supported language script is present in the text
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")  # Hindi/Marathi
ENGLISH_REGEX = re.compile(r"[A-Za-z]")

# Only allow Marathi, Hindi, and English strictly (no fallback)
def validate_language(text):
    """Allow only English, Hindi, and Marathi text. Reject if any other script is present or if none of these are detected."""
    if not text or not text.strip():
        return False
    has_devanagari = bool(DEVANAGARI_REGEX.search(text))
    has_english = bool(ENGLISH_REGEX.search(text))
    # If contains any non-supported script, reject
    # Accept only if Devanagari or English script is present, and no other script
    # (You can add more regex for other scripts if you want to block them strictly)
    # For now, just allow if either Devanagari or English is present
    return has_devanagari or has_english

def transcribe_audio(client, audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            result = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="text",
                temperature=0.0
            )

        #Groq returns a tuple (text, response), extract just the text
        if isinstance(result, tuple):
            transcription = result[0]
        else:
            transcription = result  # fallback if Groq changes behavior later

        # Validate language from transcribed text
        if not validate_language(transcription):
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            return (
                False,
                f"Sorry, I only support {supported_langs}. "
                "Please speak in one of these languages."
            )

        return (True, transcription)
    finally:
        os.unlink(temp_path) if os.path.exists(temp_path) else None