"""
Camb.ai TTS: synthesize speech, cache WAV bytes, play audio.
See https://docs.camb.ai/sdk-guides/python-sdk
"""
import os
import sys
import tempfile
import threading
from dotenv import load_dotenv

load_dotenv()

_cache = {}
_cache_lock = threading.Lock()

CAMB_VOICE_ID = 147320
CAMB_LANGUAGE = "en-us"
CAMB_SPEECH_MODEL = "mars-flash"


def _normalize_cache_key(text: str) -> str:
    return text.strip().lower()


def get_last_segment(buffer: str) -> str:
    """Last whitespace-separated segment (current word being typed)."""
    parts = buffer.strip().split()
    return parts[-1] if parts else ""


def letter_count(segment: str) -> int:
    return len([c for c in segment if c.isalpha()])


def get_or_synthesize_wav(text: str) -> bytes:
    """Return WAV bytes for text; use cache or Camb API."""
    key = _normalize_cache_key(text)
    if len(key) < 3:
        raise ValueError("TTS text must be at least 3 characters")
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    api_key = (os.getenv("CAMB_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set CAMB_API_KEY in .env")
    from camb.client import CambAI
    from camb.types import StreamTtsOutputConfiguration

    client = CambAI(api_key=api_key)
    chunks = []
    for chunk in client.text_to_speech.tts(
        text=text.strip(),
        language=CAMB_LANGUAGE,
        voice_id=CAMB_VOICE_ID,
        speech_model=CAMB_SPEECH_MODEL,
        output_configuration=StreamTtsOutputConfiguration(format="wav"),
    ):
        chunks.append(chunk)
    data = b"".join(chunks)
    # Debug aid: confirm payload shape and optionally save one file.
    print(f"TTS bytes={len(data)} header={data[:12]!r}")
    if os.getenv("CAMB_TTS_DEBUG_SAVE", "0") == "1":
        with open("debug_camb.wav", "wb") as dbg:
            dbg.write(data)
        print("Saved debug_camb.wav")
    with _cache_lock:
        _cache[key] = data
    return data


def play_wav_bytes(data: bytes) -> None:
    """Play WAV bytes (blocking). Prefer pygame; fallback to winsound on Windows."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            return
        except Exception:
            pass

        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        else:
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["afplay", path], check=False)
            else:
                subprocess.run(["aplay", path], check=False)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
