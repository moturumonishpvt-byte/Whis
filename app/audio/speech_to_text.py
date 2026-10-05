"""Speech-to-text (STT) interface for WHIS.

Provides local, offline speech transcription via available backends.

Current backend availability:
  - openai-whisper : NOT installed in this environment
  - vosk           : NOT installed in this environment

When no local backend is available, transcribe() returns a
SpeechResult(success=False, backend=None) with a clear limitation
message. No false transcription is ever claimed.

NO always-on microphone.
NO wake-word service.
NO background audio recording.

All audio access is EXPLICIT: the caller provides a pre-recorded audio
file. WHIS never opens the microphone autonomously.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Known backends to attempt, in order of preference
_STT_BACKENDS = ["whisper", "vosk"]


class STTError(Exception):
    """Raised when STT configuration or backend is unavailable."""


@dataclass(frozen=True)
class SpeechResult:
    """Structured result from a speech-to-text transcription attempt."""

    success: bool
    text: Optional[str] = None
    backend: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    limitation: Optional[str] = None


def _detect_available_backend() -> Optional[str]:
    """Detect the first available local STT backend.

    Returns:
        Backend name string if found, None if no backend is available.
    """
    for backend in _STT_BACKENDS:
        try:
            __import__(backend)
            return backend
        except ImportError:
            continue
    return None


def _get_wav_duration(audio_path: Path) -> Optional[float]:
    """Return duration of a WAV file in seconds, or None if unreadable."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else None
    except Exception:
        return None


def _validate_audio_path(audio_path: str) -> Path:
    """Validate that an audio file exists and is a supported format.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Resolved Path to the audio file.

    Raises:
        STTError: If the path is invalid or file does not exist.
    """
    if not audio_path or not str(audio_path).strip():
        raise STTError("Audio path cannot be empty.")
    if "\x00" in audio_path:
        raise STTError("Audio path contains null bytes.")

    resolved = Path(audio_path).resolve()
    if not resolved.exists():
        raise STTError(f"Audio file not found: {resolved}")
    if not resolved.is_file():
        raise STTError(f"Path is not a file: {resolved}")
    if resolved.stat().st_size == 0:
        raise STTError(f"Audio file is empty: {resolved}")

    supported = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    ext = resolved.suffix.lower()
    if ext not in supported:
        raise STTError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(supported)}"
        )

    return resolved


def transcribe(audio_path: str, backend: Optional[str] = None) -> SpeechResult:
    """Transcribe speech from a pre-recorded audio file.

    All audio access is EXPLICIT — the caller provides a file path.
    WHIS never autonomously opens a microphone.

    Args:
        audio_path: Path to a pre-recorded audio file (.wav recommended).
        backend: Override backend ("whisper", "vosk"). None = auto-detect.

    Returns:
        SpeechResult with transcribed text, or error/limitation information.

    Limitation:
        If no local offline STT backend is installed, returns
        SpeechResult(success=False) with a documented limitation message.
        No false transcription is produced.
    """
    # Validate audio file
    try:
        resolved = _validate_audio_path(audio_path)
    except STTError as exc:
        return SpeechResult(success=False, error=str(exc))

    duration = _get_wav_duration(resolved)

    # Detect backend
    active_backend = backend or _detect_available_backend()

    if active_backend is None:
        limitation = (
            "No local offline STT backend is installed in this environment. "
            "Install 'openai-whisper' (pip install openai-whisper) for CPU-based "
            "offline transcription, or 'vosk' for a lighter alternative. "
            "No cloud STT is used by WHIS."
        )
        logger.warning("STT: No backend available. %s", limitation)
        return SpeechResult(
            success=False,
            backend=None,
            duration_seconds=duration,
            limitation=limitation,
            error="No local STT backend available.",
        )

    # --- whisper backend ---
    if active_backend == "whisper":
        try:
            import whisper as _whisper  # type: ignore

            logger.info("STT: transcribing with whisper (file: %s).", resolved)
            model = _whisper.load_model("base")
            result = model.transcribe(str(resolved))
            text = result.get("text", "").strip()
            return SpeechResult(
                success=True,
                text=text,
                backend="whisper",
                duration_seconds=duration,
            )
        except Exception as exc:
            logger.error("STT whisper transcription failed: %s", exc)
            return SpeechResult(
                success=False,
                backend="whisper",
                duration_seconds=duration,
                error=str(exc),
            )

    # --- vosk backend ---
    if active_backend == "vosk":
        try:
            import json as _json
            import vosk as _vosk  # type: ignore

            logger.info("STT: transcribing with vosk (file: %s).", resolved)
            _vosk.SetLogLevel(-1)

            # Look for a vosk model in the models directory
            from pathlib import Path as _P
            vosk_model_dirs = list(_P("models").glob("vosk*"))
            if not vosk_model_dirs:
                return SpeechResult(
                    success=False,
                    backend="vosk",
                    duration_seconds=duration,
                    error="Vosk model not found. Download a vosk model to models/vosk-<lang>/.",
                    limitation="Vosk is installed but no model directory was found.",
                )

            model = _vosk.Model(str(vosk_model_dirs[0]))

            with wave.open(str(resolved), "rb") as wf:
                rec = _vosk.KaldiRecognizer(model, wf.getframerate())
                chunk = wf.readframes(4000)
                while chunk:
                    if rec.AcceptWaveform(chunk):
                        r = _json.loads(rec.Result())
                        text_parts.append(r.get("text", ""))
                    chunk = wf.readframes(4000)
                final_r = _json.loads(rec.FinalResult())
                text_parts.append(final_r.get("text", ""))
                text = " ".join(p for p in text_parts if p).strip()

            return SpeechResult(
                success=True,
                text=text,
                backend="vosk",
                duration_seconds=duration,
            )

        except Exception as exc:
            logger.error("STT vosk transcription failed: %s", exc)
            return SpeechResult(
                success=False,
                backend="vosk",
                duration_seconds=duration,
                error=str(exc),
            )

    return SpeechResult(
        success=False,
        error=f"Unknown or unsupported STT backend: '{active_backend}'.",
    )
