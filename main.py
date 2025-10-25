# pip install -U assemblyai pyaudio python-dotenv
# PortAudio system lib (needed for PyAudio / SoundDevice)
# brew install portaudio
# python -m pip install --upgrade pip setuptools wheel
# Install AssemblyAI extras (brings PyAudio and/or sounddevice)
# pip install "assemblyai[extras]" --no-cache-dir
# System prereqs: ffmpeg is recommended for AV handling
# macOS (brew):
# brew install ffmpeg
# Create/activate venv as you like, then:
# pip install --upgrade pip setuptools wheel
# pip install streamlit streamlit-webrtc assemblyai aiortc av scipy numpy python-dotenv

from __future__ import annotations

import os
import logging
from typing import Optional
from dotenv import load_dotenv

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

load_dotenv()
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing API key. Set ASSEMBLYAI_API_KEY in your environment.")
client = StreamingClient(StreamingClientOptions(api_key=api_key))
# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000

def on_begin(client: StreamingClient, event: BeginEvent) -> None:
    logger.info(f"Session started: {event.id}")

def on_turn(client: StreamingClient, event: TurnEvent) -> None:
    # `event.transcript` is partial/final text; `end_of_turn` marks a final segment
    print(f"{event.transcript} (final={event.end_of_turn})")
    # Example: enable formatted turns once
    if event.end_of_turn and not event.turn_is_formatted:
        client.set_params(StreamingSessionParameters(format_turns=True))

def on_terminated(client: StreamingClient, event: TerminationEvent) -> None:
    logger.info(f"Session terminated: processed {event.audio_duration_seconds:.2f}s of audio")

def on_error(client: StreamingClient, error: StreamingError) -> None:
    logger.error(f"Streaming error: {error}")

def main() -> None:

    # Register event handlers
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # Connect the streaming session
    client.connect(
        StreamingParameters(
            sample_rate=AUDIO_SAMPLE_RATE,
            format_turns=True,   # start formatted if you like
        )
    )

    try:
        # Stream the microphone
        mic_stream = aai.extras.MicrophoneStream(sample_rate=AUDIO_SAMPLE_RATE)
        print(mic_stream)
        client.stream(mic_stream)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # Politely end the session, sending a termination signal to get final stats
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
