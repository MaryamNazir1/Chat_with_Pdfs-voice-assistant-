import streamlit as st
import base64
import os
import wave
from dotenv import load_dotenv
from deepgram import DeepgramClient, SpeakOptions
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize the Groq client with the API key
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Filename for the output audio file
FILENAME = "output.wav"

def autoplay_audio(file_path: str):
    """Plays audio file automatically in the Streamlit app."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")

        # Create the HTML for the audio player
        audio_player = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        # Display the audio player in the Streamlit app
        st.markdown(audio_player, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error playing audio: {e}")

def generate_speech(text_input):
    """Generates speech from text and saves it to a file."""
    try:
        SPEAK_OPTIONS = {"text": text_input}
        deepgram_client = DeepgramClient(api_key=os.getenv("DG_API_KEY"))

        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )

        deepgram_client.speak.v("1").save(FILENAME, SPEAK_OPTIONS, options)
        return FILENAME

    except Exception as e:
        st.error(f"Exception during speech generation: {e}")
        return None

def transcribe_audio(file_path, prompt=""):
    """Transcribes the audio file to text using the Groq API."""
    try:
        with open(file_path, "rb") as file:
            translation = groq_client.audio.translations.create(
                file=(file_path, file.read()),  # Required audio file
                model="whisper-large-v3",  # Required model to use for translation
                prompt=prompt,  # Optional
                response_format="json",  # Optional
                temperature=0.0  # Optional
            )
            
        # Return the transcription text
        return translation.text if hasattr(translation, 'text') else "Transcription failed."

    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "An error occurred during transcription."