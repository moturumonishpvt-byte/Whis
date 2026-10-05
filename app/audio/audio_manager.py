"""AudioManager — unified API for WHIS speech-to-text and text-to-speech.

Usage:
    from app.audio import AudioManager

    audio = AudioManager()

    # Transcribe a pre-recorded audio file (explicit, no live mic)
    result = audio.transcribe("recording.wav")

    # Speak text via Windows SAPI (local, no cloud)
    result = audio.speak("Hello, I am WHIS.")

NO always-on microphone.
NO wake-word detection.
NO background audio recording.
NO cloud audio upload.

All audio input is EXPLICIT: the caller provides a pre-recorded file.
All audio output is EXPLICIT: the caller calls speak().
"""

from __future__ import annotations

import logging
from typing import Optional

from app.audio.speech_to_text import SpeechResult, transcribe
from app.audio.text_to_speech import SpeechSynthesisResult, speak

logger = logging.getLogger(__name__)


class AudioManager:
    """Unified audio API for WHIS.

    Wraps speech-to-text and text-to-speech in a single interface.
    All calls are explicit on-demand. No background audio threads.
    """

    def transcribe(
        self,
        audio_path: str,
        backend: Optional[str] = None,
    ) -> SpeechResult:
        """Transcribe speech from a pre-recorded audio file.

        The caller provides a file path. WHIS never opens a microphone.

        Args:
            audio_path: Path to an audio file (.wav, .mp3, .ogg, etc.).
            backend: Force a specific STT backend ("whisper", "vosk").
                     None = auto-detect best available backend.

        Returns:
            SpeechResult with transcribed text, or limitation/error info
            if no local STT backend is installed.
        """
        logger.debug("AudioManager: transcribe called for '%s'.", audio_path)
        return transcribe(audio_path=audio_path, backend=backend)

    def speak(
        self,
        text: str,
        output_path: Optional[str] = None,
        rate: int = 0,
        volume: int = 100,
    ) -> SpeechSynthesisResult:
        """Synthesize speech from text using Windows SAPI (local, no cloud).

        Args:
            text: Text to synthesize.
            output_path: Optional WAV file path. If None, plays through speakers.
            rate: Speech rate (-10 to 10, 0 = normal).
            volume: Volume level (0–100).

        Returns:
            SpeechSynthesisResult with success flag and error info if any.
        """
        logger.debug("AudioManager: speak called (%d chars).", len(text))
        return speak(
            text=text,
            output_path=output_path,
            rate=rate,
            volume=volume,
        )
