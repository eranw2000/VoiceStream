import os
import numpy as np
from dotenv import load_dotenv
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Generate a test tone at 440Hz (A note) for 3 seconds
SAMPLE_RATE = 16000
duration = 3  # seconds
t = np.linspace(0, duration, SAMPLE_RATE * duration, False)
tone = np.sin(2 * np.pi * 440 * t)

# Scale to int16 range with good amplitude
audio_data = (tone * 10000).astype(np.int16)

print(f"Generated {len(audio_data)} samples, RMS={np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.1f}")

client = StreamingClient(StreamingClientOptions(api_key=API_KEY))

turn_received = False

def on_begin(_, event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(_, event: TurnEvent):
    global turn_received
    turn_received = True
    print(f"TURN EVENT: transcript='{event.transcript}', end_of_turn={event.end_of_turn}")

def on_terminated(_, event: TerminationEvent):
    print(f"Session ended: {event.audio_duration_seconds:.2f}s")

def on_error(_, error: StreamingError):
    print(f"Error: {error}")

client.on(StreamingEvents.Begin, on_begin)
client.on(StreamingEvents.Turn, on_turn)
client.on(StreamingEvents.Termination, on_terminated)
client.on(StreamingEvents.Error, on_error)

client.connect(StreamingParameters(sample_rate=SAMPLE_RATE, format_turns=True))

def audio_generator():
    chunk_size = 3200  # 100ms chunks
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        yield chunk.tobytes()

print("Streaming test tone to AssemblyAI...")
client.stream(audio_generator())
print(f"Streaming finished. Turn received: {turn_received}")
