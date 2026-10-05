"""Text-to-speech (TTS) interface for WHIS.

Provides local speech synthesis using the Windows Speech API (SAPI)
via PowerShell. No third-party libraries required. No cloud TTS.

Usage:
    from app.audio.text_to_speech import speak

    result = speak("Hello, I am WHIS.")

Features:
  - Uses Windows SAPI (System.Speech) — always available on Windows 10/11
  - No microphone access
  - No cloud communication
  - Shell injection protection via argument list (shell execution disabled)
  - Optional output to WAV file via SAPI SpeechAudioFormatInfo

Limitation:
  - WAV export requires a compatible SAPI voice; quality depends on
    installed Windows voices.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum safe text length for a single TTS call (characters)
_MAX_TEXT_LENGTH = 4096


class TTSError(Exception):
    """Raised when TTS synthesis fails."""


@dataclass(frozen=True)
class SpeechSynthesisResult:
    """Structured result from a text-to-speech synthesis call."""

    success: bool
    text: Optional[str] = None
    backend: str = "windows_sapi"
    output_path: Optional[str] = None
    error: Optional[str] = None


def _sanitize_text_for_ps(text: str) -> str:
    """Escape text for safe embedding in a PowerShell string argument.

    Replaces characters that could break the PS -Command argument boundary.
    """
    # Escape single-quotes (PowerShell string delimiters) by doubling them
    return text.replace("'", "''").replace("\x00", "").replace("\r", " ")


def speak(
    text: str,
    output_path: Optional[str] = None,
    rate: int = 0,
    volume: int = 100,
) -> SpeechSynthesisResult:
    """Synthesize speech from text using Windows SAPI.

    This uses System.Speech via PowerShell (subprocess, shell=False).
    No microphone access. No cloud calls. No always-on audio.

    Args:
        text: Text to synthesize. Maximum 4096 characters.
        output_path: Optional WAV file path to save audio to.
                     If None, audio is played directly through the speakers.
        rate: Speech rate (-10 to 10, 0 = normal).
        volume: Volume (0–100).

    Returns:
        SpeechSynthesisResult with success flag and any error information.
    """
    if not text or not text.strip():
        return SpeechSynthesisResult(
            success=False,
            error="Text cannot be empty.",
        )

    if len(text) > _MAX_TEXT_LENGTH:
        return SpeechSynthesisResult(
            success=False,
            error=f"Text length {len(text)} exceeds maximum of {_MAX_TEXT_LENGTH} characters.",
        )

    # Clamp rate and volume
    rate = max(-10, min(10, rate))
    volume = max(0, min(100, volume))

    safe_text = _sanitize_text_for_ps(text)

    if output_path is not None:
        # Save to WAV file via SAPI SpeechFileStream
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest_str = str(dest).replace("'", "''")

        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; "
            f"$s.Volume = {volume}; "
            f"$s.SetOutputToWaveFile('{dest_str}'); "
            f"$s.Speak('{safe_text}'); "
            "$s.SetOutputToDefaultAudioDevice(); "
            "$s.Dispose();"
        )

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip() or "SAPI error."
                logger.error("TTS WAV export failed: %s", err)
                return SpeechSynthesisResult(
                    success=False,
                    text=text,
                    error=f"TTS WAV export failed: {err}",
                )

            if not dest.exists() or dest.stat().st_size == 0:
                return SpeechSynthesisResult(
                    success=False,
                    text=text,
                    error="TTS produced no output file.",
                )

            logger.info("TTS: synthesized %d chars to WAV: %s", len(text), dest)
            return SpeechSynthesisResult(
                success=True,
                text=text,
                backend="windows_sapi",
                output_path=str(dest),
            )

        except subprocess.TimeoutExpired:
            return SpeechSynthesisResult(
                success=False,
                text=text,
                error="TTS synthesis timed out after 30s.",
            )
        except Exception as exc:
            logger.error("TTS WAV export raised unexpected error: %s", exc)
            return SpeechSynthesisResult(success=False, text=text, error=str(exc))

    else:
        # Speak directly through speakers
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; "
            f"$s.Volume = {volume}; "
            f"$s.Speak('{safe_text}'); "
            "$s.Dispose();"
        )

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip() or "SAPI error."
                logger.error("TTS speak failed: %s", err)
                return SpeechSynthesisResult(
                    success=False,
                    text=text,
                    error=f"TTS speak failed: {err}",
                )

            logger.info("TTS: synthesized and played %d chars via SAPI.", len(text))
            return SpeechSynthesisResult(
                success=True,
                text=text,
                backend="windows_sapi",
            )

        except subprocess.TimeoutExpired:
            return SpeechSynthesisResult(
                success=False,
                text=text,
                error="TTS synthesis timed out after 60s.",
            )
        except Exception as exc:
            logger.error("TTS speak raised unexpected error: %s", exc)
            return SpeechSynthesisResult(success=False, text=text, error=str(exc))
