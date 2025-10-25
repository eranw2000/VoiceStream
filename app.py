#!/usr/bin/env python3
"""Simple Streamlit app for real-time speech-to-text using AssemblyAI"""
import os
import queue
import threading
import logging
from dotenv import load_dotenv

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
import numpy as np
from scipy.signal import resample_poly

from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
    BeginEvent,
    TerminationEvent,
    StreamingError,
)

# ------------------ Setup ------------------
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
TARGET_SR = 16000

logging.basicConfig(level=logging.ERROR)  # Only show errors
log = logging.getLogger("voiceapp")
log.setLevel(logging.INFO)  # But keep INFO for our app

# Suppress noisy warnings from streamlit_webrtc
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)

st.set_page_config(page_title="Talk-to-Text", page_icon="🎙️", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Make the app look cleaner */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Session State ------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
    log.info("Initialized transcript in session state")
if "interim_text" not in st.session_state:
    st.session_state.interim_text = ""
if "last_final" not in st.session_state:
    st.session_state.last_final = ""  # Track last final to avoid duplicates
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "text_queue" not in st.session_state:
    st.session_state.text_queue = queue.Queue()
    log.info("Initialized text_queue in session state")
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "aai_thread" not in st.session_state:
    st.session_state.aai_thread = None

# ------------------ Audio Processing ------------------
def convert_audio(frame: av.AudioFrame) -> bytes:
    """Convert browser audio to PCM16 mono @ 16kHz for AssemblyAI - use PyAV's resampler"""
    import av.audio.resampler

    # Use PyAV's built-in audio resampler (handles stereo->mono and resampling properly)
    resampler = av.audio.resampler.AudioResampler(
        format='s16',  # 16-bit signed integer (PCM16)
        layout='mono',  # Convert to mono
        rate=TARGET_SR  # Resample to 16kHz
    )

    # Resample the frame
    resampled_frames = resampler.resample(frame)

    # Get the audio data as bytes
    if resampled_frames:
        audio_data = resampled_frames[0].to_ndarray()

        # Let browser's autoGainControl handle the levels
        return audio_data.tobytes()
    else:
        # Return silence if no frames (shouldn't happen)
        return b'\x00' * 3200

class AudioProcessor:
    def __init__(self, audio_queue):
        self.buffer = b""
        self.chunk_size = 3200  # 100ms @ 16kHz
        self.frame_count = 0
        self.audio_queue = audio_queue

    async def recv_queued(self, frames):
        """Process batched audio frames"""
        for frame in frames:
            self.frame_count += 1

            pcm = convert_audio(frame)
            self.buffer += pcm

            # Send 100ms chunks to AssemblyAI
            while len(self.buffer) >= self.chunk_size:
                chunk = self.buffer[:self.chunk_size]
                self.buffer = self.buffer[self.chunk_size:]
                self.audio_queue.put(chunk)

        return frames

# ------------------ AssemblyAI Streaming ------------------
def run_assemblyai(audio_queue, text_queue, stop_event):
    """Background thread that streams audio to AssemblyAI"""
    log.info("AssemblyAI thread starting...")

    client = StreamingClient(StreamingClientOptions(api_key=API_KEY))

    def on_begin(_, event: BeginEvent):
        log.info(f"Session started: {event.id}")

    def on_turn(_, event: TurnEvent):
        # Send both interim and final transcripts
        # Format: (transcript, is_final)
        if event.transcript:
            text_queue.put((event.transcript, event.end_of_turn))

    def on_terminated(_, event: TerminationEvent):
        log.info(f"Session ended: {event.audio_duration_seconds:.2f}s")

    def on_error(_, error: StreamingError):
        log.error(f"Error: {error}")

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # Connect and stream
    client.connect(StreamingParameters(
        sample_rate=TARGET_SR,
        format_turns=True,
    ))
    log.info("Connected to AssemblyAI, starting to stream...")

    def audio_generator():
        chunk_count = 0
        empty_chunk = b'\x00' * 3200  # Silence chunk
        all_audio = []  # Collect audio for debugging

        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
                chunk_count += 1
                all_audio.append(chunk)

                # Log first chunk only
                if chunk_count == 1:
                    log.info("Audio streaming started")
                yield chunk
            except queue.Empty:
                # Send silence to keep connection alive
                yield empty_chunk
                continue

        log.info(f"Audio stream ended. Total chunks: {chunk_count}")

    client.stream(audio_generator())
    log.info("AssemblyAI thread finished")

# ------------------ UI ------------------
st.title("🎙️ Talk-to-Text")

if not API_KEY:
    st.error("Missing ASSEMBLYAI_API_KEY in .env file")
    st.stop()

# Only auto-refresh when streaming to update transcript
if st.session_state.streaming:
    count = st_autorefresh(interval=100, key="refresh")
else:
    count = 0

# Voice/Chat input field with microphone button
if not st.session_state.streaming:
    # Show editable input with previous transcript or placeholder
    col1, col2 = st.columns([9, 1])
    with col1:
        current_value = st.session_state.transcript.strip() if st.session_state.transcript.strip() else ""
        user_input = st.text_input("Chat Input", value=current_value, placeholder="Ask anything", key="chat_editable", label_visibility="collapsed")

        # If user edited the text, update the transcript
        if user_input != current_value:
            st.session_state.transcript = user_input
    with col2:
        if st.button("🎤", key="mic_btn", help="Start voice recording"):
            st.session_state.streaming = True
            # Don't clear transcript - append to it
            st.session_state.interim_text = ""
            st.session_state.last_final = ""
            # Clear queues
            while not st.session_state.audio_queue.empty():
                st.session_state.audio_queue.get()
            while not st.session_state.text_queue.empty():
                st.session_state.text_queue.get()
            # Clear stop event
            st.session_state.stop_event.clear()
            # Start AssemblyAI thread
            thread = threading.Thread(
                target=run_assemblyai,
                args=(st.session_state.audio_queue, st.session_state.text_queue, st.session_state.stop_event),
                daemon=True
            )
            thread.start()
            st.session_state.aai_thread = thread
            log.info("Voice recording started")
            st.rerun()
else:
    # Show transcript in the input field while recording
    col1, col2 = st.columns([9, 1])
    with col1:
        # Combine final transcript with interim text
        current_text = st.session_state.transcript.strip()
        if st.session_state.interim_text:
            current_text += " " + st.session_state.interim_text

        st.text_input("Live Transcript", value=current_text, key=f"live_transcript_{count}", disabled=True, label_visibility="collapsed")
    with col2:
        if st.button("⏹️", key="stop_btn", help="Stop recording"):
            st.session_state.streaming = False
            st.session_state.stop_event.set()
            # Keep the transcript for editing
            log.info(f"Voice recording stopped. Transcript: '{st.session_state.transcript.strip()}'")
            st.rerun()

# Update transcript from queue
while not st.session_state.text_queue.empty():
    text, is_final = st.session_state.text_queue.get()
    if is_final:
        # Simple duplicate check - only skip if exactly the same as last final
        text_clean = text.lower().strip().rstrip('.').rstrip(',').rstrip('!').rstrip('?')
        last_clean = st.session_state.last_final.lower().strip().rstrip('.').rstrip(',').rstrip('!').rstrip('?')

        # Only skip exact duplicates (like "hello" and "Hello.")
        if text_clean and text_clean != last_clean:
            # Append final transcript
            st.session_state.transcript += text + " "
            st.session_state.last_final = text
            log.info(f"Added: '{text}'")
        else:
            log.info(f"Skipped exact duplicate: '{text}'")

        st.session_state.interim_text = ""
    else:
        # Show interim result (but don't log every interim update)
        st.session_state.interim_text = text

# WebRTC audio streamer (run in background without UI)
# Note: The WebRTC component will still show some UI, but we minimize it
with st.expander("Audio Settings (Advanced)", expanded=False):
    audio_queue = st.session_state.audio_queue
    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: AudioProcessor(audio_queue),
        media_stream_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            },
            "video": False
        },
        async_processing=True,
        desired_playing_state=st.session_state.streaming,
    )
