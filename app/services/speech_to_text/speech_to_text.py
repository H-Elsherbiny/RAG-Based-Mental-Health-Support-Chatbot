import os
from typing import Dict, Any
from openai import OpenAI
from groq import Groq 
from app.core.config import GROQ_API_KEY
from app.core.config import OPENAI_API_KEY
from .config import WHISPER_LANG_MAPPING

class SpeechToText:
    def __init__(self):
        # Initializes the client using the standard GROQ_API_KEY env variable
        # If you are routing Groq models through Groq or another provider, 
        # ensure your base_url and api_key point to the correct endpoint.
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "whisper-large-v3"

    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribes an audio file using Groq's Whisper API and captures 
        the detected source language.
        
        Args:
            audio_file_path (str): Path to the local audio file (.wav, .mp3, .m4a, etc.)
            
        Returns:
            Dict[str, Any]: A dictionary containing 'text' and 'detected_language'
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")

        with open(audio_file_path, "rb") as audio_file:
            # We use transcriptions.create with response_format="verbose_json"
            # to extract the language property directly from Whisper.
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json"
            )
            
        return {
            "text": response.text,
            "detected_language_code": response.language, # Returns ISO-639-1 code
            "detected_language_name": WHISPER_LANG_MAPPING.get(response.language, "Unknown")
        }