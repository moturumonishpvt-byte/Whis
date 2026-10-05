"""WHIS audio package.

Exports:
    AudioManager  — unified STT/TTS API
    transcribe    — function shortcut for STT
    speak         — function shortcut for TTS
"""

from app.audio.audio_manager import AudioManager
from app.audio.speech_to_text import SpeechResult, transcribe
from app.audio.text_to_speech import SpeechSynthesisResult, speak

__all__ = [
    "AudioManager",
    "SpeechResult",
    "SpeechSynthesisResult",
    "speak",
    "transcribe",
]
