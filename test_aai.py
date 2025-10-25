#!/usr/bin/env python3
"""Simple test to verify AssemblyAI v3 streaming works"""
import os
import time
from dotenv import load_dotenv
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("AssemblyAI_KEY")

print(f"API Key present: {bool(API_KEY)}")

def audio_generator():
    """Generate 5 seconds of silence (just zeros)"""
    # 16kHz, 16-bit PCM, mono
    # Send 100ms chunks (1600 samples * 2 bytes = 3200 bytes)
    chunk_size = 3200
    chunk = b'\x00' * chunk_size

    for i in range(50):  # 50 chunks = 5 seconds
        print(f"Sending chunk {i+1}/50")
        yield chunk
        time.sleep(0.1)  # 100ms between chunks
    print("Done sending audio")

client = StreamingClient(StreamingClientOptions(api_key=API_KEY))

def on_begin(_, e):
    print(f"BEGIN: Session {e.id}")

def on_turn(_, e):
    print(f"TURN: transcript='{e.transcript}', end_of_turn={e.end_of_turn}, words={len(e.words)}")

def on_term(_, e):
    print(f"TERMINATION: {e.audio_duration_seconds}s")

def on_error(_, e):
    print(f"ERROR: {e}")

client.on(StreamingEvents.Begin, on_begin)
client.on(StreamingEvents.Turn, on_turn)
client.on(StreamingEvents.Termination, on_term)
client.on(StreamingEvents.Error, on_error)

print("Connecting...")
client.connect(StreamingParameters(
    sample_rate=16000,
    format_turns=True,
    max_turn_silence=500,
))

print("Streaming...")
client.stream(audio_generator())
print("Finished")
