import os
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

transcriber = aai.Transcriber()

print("Testing batch transcription on debug_audio.wav...")
transcript = transcriber.transcribe("debug_audio.wav")

print(f"\nStatus: {transcript.status}")
print(f"Text: {transcript.text}")
print(f"Confidence: {transcript.confidence if hasattr(transcript, 'confidence') else 'N/A'}")
print(f"Audio duration: {transcript.audio_duration if hasattr(transcript, 'audio_duration') else 'N/A'}")

if transcript.error:
    print(f"Error: {transcript.error}")
