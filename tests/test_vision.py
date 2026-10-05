"""Stage 7 — Vision and Audio tests.

Covers:
  - screen capture (real + invalid inputs)
  - region capture
  - camera API (explicit-only contract)
  - image validation
  - VisionEngine API surface
  - OCR engine selection logic
  - GLM-OCR primary path (mocked inference)
  - DeepSeek-OCR fallback path (mocked inference)
  - OCR failure handling (both engines fail)
  - unknown OCR engine rejection
  - STT interface (no backend available = honest limitation)
  - STT invalid audio path
  - TTS interface (mocked PS call)
  - TTS empty text rejection
  - TTS text too long rejection
  - no background microphone assertion
  - no continuous camera loop assertion
  - no subprocess outside RuntimeAdapter in vision/ocr/audio
  - model switching compatibility (ImageResult fields)
  - structured result validation (all result types)
  - image format validation (supported/unsupported)
  - null-byte path rejection
  - empty path rejection
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_bmp(path: str, width: int = 4, height: int = 4) -> None:
    """Write a minimal valid 24-bit BMP file to path."""
    row_bytes = ((width * 24 + 31) // 32) * 4
    image_size = row_bytes * height
    bmi_size = 40
    pixel_offset = 14 + bmi_size
    file_size = pixel_offset + image_size

    bmp_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, pixel_offset)
    bmi = struct.pack(
        "<IiiHHIIIIII",
        bmi_size, width, -height, 1, 24, 0, image_size, 0, 0, 0, 0,
    )
    pixel_data = bytes(image_size)

    with open(path, "wb") as f:
        f.write(bmp_header)
        f.write(bmi)
        f.write(pixel_data)


def _make_minimal_wav(path: str, duration_frames: int = 1600) -> None:
    """Write a minimal valid WAV file (silence, 16kHz mono, 16-bit)."""
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * duration_frames * 2)


# ---------------------------------------------------------------------------
# 1. Screen Capture Tests
# ---------------------------------------------------------------------------

class TestScreenCapture(unittest.TestCase):

    def test_capture_screen_returns_capture_result(self):
        """capture_screen() must return a CaptureResult regardless of success."""
        from app.vision.screen import capture_screen, CaptureResult

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "screen.bmp")
            result = capture_screen(output_path=out)
            self.assertIsInstance(result, CaptureResult)

    def test_capture_screen_success_has_image_path(self):
        """Successful screen capture must have a non-empty image_path."""
        from app.vision.screen import capture_screen

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "screen.bmp")
            result = capture_screen(output_path=out)
            if result.success:
                self.assertIsNotNone(result.image_path)
                self.assertTrue(Path(result.image_path).exists())
                self.assertGreater(result.width, 0)
                self.assertGreater(result.height, 0)
            else:
                # On headless CI-like environments it may fail gracefully
                self.assertIsNotNone(result.error)

    def test_capture_screen_default_path_creates_file(self):
        """capture_screen() with no output_path must create a temp file."""
        from app.vision.screen import capture_screen

        result = capture_screen()
        if result.success:
            self.assertIsNotNone(result.image_path)
            p = Path(result.image_path)
            self.assertTrue(p.exists(), "Expected temp BMP to exist.")
            p.unlink(missing_ok=True)  # clean up

    def test_capture_region_returns_capture_result(self):
        """capture_region() must return CaptureResult."""
        from app.vision.screen import capture_region, CaptureResult

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "region.bmp")
            result = capture_region(0, 0, 100, 100, output_path=out)
            self.assertIsInstance(result, CaptureResult)

    def test_capture_region_invalid_dimensions_fails(self):
        """Negative or zero region dimensions must return success=False."""
        from app.vision.screen import capture_region

        result = capture_region(0, 0, -10, 100)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

        result = capture_region(0, 0, 100, 0)
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# 2. Camera API Tests
# ---------------------------------------------------------------------------

class TestCameraCapture(unittest.TestCase):

    def test_capture_camera_frame_returns_capture_result(self):
        """capture_camera_frame() must always return a CaptureResult."""
        from app.vision.camera import capture_camera_frame, CaptureResult

        result = capture_camera_frame()
        self.assertIsInstance(result, CaptureResult)

    def test_camera_no_continuous_loop(self):
        """camera.py must not define any continuous/background capture loop."""
        import inspect
        import app.vision.camera as cam_module

        src = inspect.getsource(cam_module)
        self.assertNotIn("while True", src, "camera.py must not contain an infinite loop.")
        self.assertNotIn("threading.Thread", src, "camera.py must not start background threads.")

    def test_camera_failure_has_error_message(self):
        """If camera is unavailable, error field must be set."""
        from app.vision.camera import capture_camera_frame

        # Patch subprocess to simulate no camera
        with patch("app.vision.camera.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="no camera", stdout="")
            result = capture_camera_frame()
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# 3. Image Validation Tests
# ---------------------------------------------------------------------------

class TestImageValidation(unittest.TestCase):

    def test_validate_existing_bmp(self):
        """validate_image() must accept a real BMP file."""
        from app.vision.image import validate_image

        fd, path = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        _make_minimal_bmp(path)
        try:
            result = validate_image(path)
            self.assertTrue(result.exists())
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_validate_nonexistent_path_raises(self):
        """validate_image() must raise ImageValidationError for missing file."""
        from app.vision.image import validate_image, ImageValidationError

        with self.assertRaises(ImageValidationError):
            validate_image("/nonexistent/path/image.bmp")

    def test_validate_unsupported_extension_raises(self):
        """validate_image() must raise for unsupported image formats."""
        from app.vision.image import validate_image, ImageValidationError

        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"not an image")
        try:
            with self.assertRaises(ImageValidationError):
                validate_image(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_validate_empty_path_raises(self):
        """validate_image() must raise for empty path string."""
        from app.vision.image import validate_image, ImageValidationError

        with self.assertRaises(ImageValidationError):
            validate_image("")

    def test_validate_null_byte_path_raises(self):
        """validate_image() must reject paths containing null bytes."""
        from app.vision.image import validate_image, ImageValidationError

        with self.assertRaises(ImageValidationError):
            validate_image("/some/path\x00/image.bmp")

    def test_validate_empty_file_raises(self):
        """validate_image() must raise for zero-byte files."""
        from app.vision.image import validate_image, ImageValidationError

        fd, path = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        try:
            with self.assertRaises(ImageValidationError):
                validate_image(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# 4. VisionEngine / Qwen-VL Selection Tests
# ---------------------------------------------------------------------------

class TestVisionEngine(unittest.TestCase):

    def test_vision_engine_instantiates(self):
        """VisionEngine must instantiate without error."""
        from app.vision import VisionEngine
        engine = VisionEngine()
        self.assertIsNotNone(engine)

    def test_vision_engine_has_required_methods(self):
        """VisionEngine must expose the required public API."""
        from app.vision import VisionEngine
        engine = VisionEngine()
        self.assertTrue(callable(engine.capture_screen))
        self.assertTrue(callable(engine.capture_region))
        self.assertTrue(callable(engine.capture_camera_frame))
        self.assertTrue(callable(engine.understand_image))
        self.assertTrue(callable(engine.extract_text))

    def test_understand_image_invalid_path_returns_failure(self):
        """understand_image() with bad path must return ImageResult(success=False)."""
        from app.vision import VisionEngine
        from app.vision.image import ImageResult

        engine = VisionEngine()
        result = engine.understand_image("/nonexistent/image.bmp", "What is this?")
        self.assertIsInstance(result, ImageResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_understand_image_selects_qwen_vl(self):
        """understand_image() must target the qwen3-vl-4b model."""
        from app.vision.image import VISION_MODEL_ID
        self.assertEqual(VISION_MODEL_ID, "qwen3-vl-4b")

    def test_understand_image_uses_runtime_adapter(self):
        """understand_image() must invoke runtime_adapter.generate(), not a raw subprocess."""
        from app.vision import VisionEngine
        from app.ai.model_manager import ModelRegistry

        engine = VisionEngine()

        fd, img_path = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        _make_minimal_bmp(img_path)

        try:
            mock_runtime = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.text = "A test image with a white background."
            mock_result.error = None
            mock_runtime.generate.return_value = mock_result

            mock_registry = MagicMock()
            mock_model_def = MagicMock()
            mock_model_def.is_experimental = False
            mock_registry.get.return_value = mock_model_def

            result = engine.understand_image(
                img_path,
                "What is this?",
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )

            mock_runtime.generate.assert_called_once()
            self.assertTrue(result.success)
            self.assertEqual(result.text, "A test image with a white background.")
        finally:
            if os.path.exists(img_path):
                os.unlink(img_path)


# ---------------------------------------------------------------------------
# 5. OCR Engine Tests
# ---------------------------------------------------------------------------

class TestOCREngine(unittest.TestCase):

    def _make_image(self) -> str:
        """Create a temp BMP and return its path."""
        fd, path = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        _make_minimal_bmp(path)
        return path

    def _mock_runtime(self, text: str = "EXTRACTED TEXT", success: bool = True):
        mock_runtime = MagicMock()
        mock_result = MagicMock()
        mock_result.success = success
        mock_result.text = text if success else None
        mock_result.error = None if success else "Inference failed."
        mock_runtime.generate.return_value = mock_result
        return mock_runtime

    def _mock_registry(self, is_experimental: bool = False):
        mock_registry = MagicMock()
        mock_def = MagicMock()
        mock_def.is_experimental = is_experimental
        mock_registry.get.return_value = mock_def
        return mock_registry

    def test_ocr_auto_uses_glm_primary(self):
        """auto mode must call GLM-OCR first."""
        from app.models.ocr import OCREngine, GLM_OCR_MODEL_ID

        img = self._make_image()
        try:
            mock_runtime = self._mock_runtime("Hello World")
            mock_registry = self._mock_registry(is_experimental=False)

            ocr = OCREngine(
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )
            result = ocr.extract_text(img, engine="auto")

            # registry.get should be called with GLM first
            calls = [str(c) for c in mock_registry.get.call_args_list]
            self.assertTrue(
                any(GLM_OCR_MODEL_ID in c for c in calls),
                f"GLM-OCR model not requested. Calls: {calls}",
            )
            self.assertTrue(result.success)
            self.assertEqual(result.text, "Hello World")
        finally:
            os.unlink(img)

    def test_ocr_auto_falls_back_to_deepseek(self):
        """auto mode must fall back to DeepSeek-OCR when GLM fails."""
        from app.models.ocr import OCREngine, GLM_OCR_MODEL_ID, DEEPSEEK_OCR_MODEL_ID

        img = self._make_image()
        try:
            call_count = [0]

            def get_side_effect(model_id: str):
                mock_def = MagicMock()
                mock_def.is_experimental = (model_id == DEEPSEEK_OCR_MODEL_ID)
                return mock_def

            def generate_side_effect(**kwargs):
                call_count[0] += 1
                mock_result = MagicMock()
                if call_count[0] == 1:
                    # First call (GLM) fails
                    mock_result.success = False
                    mock_result.text = None
                    mock_result.error = "GLM inference error."
                else:
                    # Second call (DeepSeek) succeeds
                    mock_result.success = True
                    mock_result.text = "DeepSeek extracted text"
                    mock_result.error = None
                return mock_result

            mock_runtime = MagicMock()
            mock_runtime.generate.side_effect = generate_side_effect
            mock_registry = MagicMock()
            mock_registry.get.side_effect = get_side_effect

            ocr = OCREngine(
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )
            result = ocr.extract_text(img, engine="auto")

            self.assertTrue(result.success)
            self.assertEqual(result.text, "DeepSeek extracted text")
            self.assertTrue(result.is_experimental)
        finally:
            os.unlink(img)

    def test_ocr_both_fail_returns_failure(self):
        """When both GLM and DeepSeek fail, result must be success=False."""
        from app.models.ocr import OCREngine

        img = self._make_image()
        try:
            mock_runtime = self._mock_runtime(success=False)
            mock_registry = self._mock_registry(is_experimental=True)

            ocr = OCREngine(
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )
            result = ocr.extract_text(img, engine="auto")
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
        finally:
            os.unlink(img)

    def test_ocr_force_glm(self):
        """engine='glm' must only call GLM-OCR."""
        from app.models.ocr import OCREngine, GLM_OCR_MODEL_ID

        img = self._make_image()
        try:
            mock_runtime = self._mock_runtime("GLM text")
            mock_registry = self._mock_registry()

            ocr = OCREngine(
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )
            result = ocr.extract_text(img, engine="glm")
            mock_registry.get.assert_called_once_with(GLM_OCR_MODEL_ID)
            self.assertTrue(result.success)
        finally:
            os.unlink(img)

    def test_ocr_force_deepseek_marks_experimental(self):
        """engine='deepseek' must mark result as is_experimental=True."""
        from app.models.ocr import OCREngine, DEEPSEEK_OCR_MODEL_ID

        img = self._make_image()
        try:
            mock_runtime = self._mock_runtime("DS text")
            mock_registry = self._mock_registry(is_experimental=True)

            ocr = OCREngine(
                runtime_adapter=mock_runtime,
                model_registry=mock_registry,
            )
            result = ocr.extract_text(img, engine="deepseek")
            mock_registry.get.assert_called_once_with(DEEPSEEK_OCR_MODEL_ID)
            self.assertTrue(result.is_experimental)
        finally:
            os.unlink(img)

    def test_ocr_unknown_engine_returns_failure(self):
        """Unknown engine name must return ImageResult(success=False)."""
        from app.models.ocr import OCREngine

        img = self._make_image()
        try:
            ocr = OCREngine()
            result = ocr.extract_text(img, engine="magic_ocr")
            self.assertFalse(result.success)
            self.assertIn("Unknown OCR engine", result.error)
        finally:
            os.unlink(img)

    def test_ocr_invalid_image_returns_failure(self):
        """Missing image must return ImageResult(success=False)."""
        from app.models.ocr import OCREngine

        ocr = OCREngine()
        result = ocr.extract_text("/nonexistent/image.bmp", engine="auto")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# 6. STT Interface Tests
# ---------------------------------------------------------------------------

class TestSpeechToText(unittest.TestCase):

    def test_transcribe_no_backend_returns_limitation(self):
        """When no STT backend is installed, return SpeechResult with limitation."""
        from app.audio.speech_to_text import transcribe, SpeechResult

        # Patch backend detection to report nothing available
        with patch("app.audio.speech_to_text._detect_available_backend", return_value=None):
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _make_minimal_wav(wav_path)
            try:
                result = transcribe(wav_path)
                self.assertIsInstance(result, SpeechResult)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.limitation)
                self.assertIsNone(result.backend)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

    def test_transcribe_invalid_path_returns_failure(self):
        """Missing audio file must return SpeechResult(success=False)."""
        from app.audio.speech_to_text import transcribe

        result = transcribe("/nonexistent/audio.wav")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_transcribe_empty_path_returns_failure(self):
        """Empty audio path must return SpeechResult(success=False)."""
        from app.audio.speech_to_text import transcribe

        result = transcribe("")
        self.assertFalse(result.success)

    def test_transcribe_unsupported_format_fails(self):
        """Unsupported audio format must return failure."""
        from app.audio.speech_to_text import transcribe

        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"not audio")
        try:
            result = transcribe(path)
            self.assertFalse(result.success)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_transcribe_never_opens_microphone(self):
        """speech_to_text module must not import any live microphone library."""
        import inspect
        import app.audio.speech_to_text as stt_module

        src = inspect.getsource(stt_module)
        self.assertNotIn("pyaudio", src.lower())
        self.assertNotIn("sounddevice", src.lower())
        self.assertNotIn("wake_word", src.lower())
        self.assertNotIn("while True", src)

    def test_no_background_microphone(self):
        """audio_manager.py must not contain always-on microphone code."""
        import inspect
        import app.audio.audio_manager as am_module

        src = inspect.getsource(am_module)
        self.assertNotIn("threading.Thread", src)
        self.assertNotIn("always_on", src)
        self.assertNotIn("wake_word", src)
        self.assertNotIn("while True", src)


# ---------------------------------------------------------------------------
# 7. TTS Interface Tests
# ---------------------------------------------------------------------------

class TestTextToSpeech(unittest.TestCase):

    def test_speak_empty_text_returns_failure(self):
        """speak() with empty text must return SpeechSynthesisResult(success=False)."""
        from app.audio.text_to_speech import speak

        result = speak("")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_speak_text_too_long_returns_failure(self):
        """speak() with text exceeding 4096 chars must return failure."""
        from app.audio.text_to_speech import speak, _MAX_TEXT_LENGTH

        long_text = "A" * (_MAX_TEXT_LENGTH + 1)
        result = speak(long_text)
        self.assertFalse(result.success)
        self.assertIn("exceeds maximum", result.error)

    def test_speak_success_via_mock(self):
        """speak() must return success=True when PowerShell returns 0."""
        from app.audio.text_to_speech import speak

        with patch("app.audio.text_to_speech.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            result = speak("Hello, I am WHIS.")

        self.assertTrue(result.success)
        self.assertEqual(result.text, "Hello, I am WHIS.")
        self.assertEqual(result.backend, "windows_sapi")

    def test_speak_failure_propagates_error(self):
        """speak() must return failure when PowerShell returns non-zero."""
        from app.audio.text_to_speech import speak

        with patch("app.audio.text_to_speech.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="SAPI error occurred.", stdout=""
            )
            result = speak("This will fail.")

        self.assertFalse(result.success)
        self.assertIn("SAPI", result.error)

    def test_speak_to_wav_file(self):
        """speak() with output_path must attempt WAV export."""
        from app.audio.text_to_speech import speak

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "output.wav")
            # Write a dummy WAV so the file-exists check passes
            _make_minimal_wav(wav_path)

            with patch("app.audio.text_to_speech.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                result = speak("Save this to a file.", output_path=wav_path)

        self.assertTrue(result.success)
        self.assertEqual(result.output_path, wav_path)

    def test_tts_no_shell_true(self):
        """text_to_speech.py must never use shell=True."""
        import inspect
        import app.audio.text_to_speech as tts_module

        src = inspect.getsource(tts_module)
        self.assertNotIn("shell=True", src)

    def test_speak_sanitizes_single_quotes(self):
        """speak() must escape single quotes to prevent PS injection."""
        from app.audio.text_to_speech import _sanitize_text_for_ps

        raw = "It's a test"
        sanitized = _sanitize_text_for_ps(raw)
        self.assertIn("''", sanitized)
        self.assertEqual(sanitized, "It''s a test")
        self.assertEqual(sanitized.count("'"), 2)


# ---------------------------------------------------------------------------
# 8. AudioManager API Tests
# ---------------------------------------------------------------------------

class TestAudioManager(unittest.TestCase):

    def test_audio_manager_instantiates(self):
        """AudioManager must instantiate without error."""
        from app.audio import AudioManager
        am = AudioManager()
        self.assertIsNotNone(am)

    def test_audio_manager_has_required_methods(self):
        """AudioManager must expose transcribe() and speak()."""
        from app.audio import AudioManager
        am = AudioManager()
        self.assertTrue(callable(am.transcribe))
        self.assertTrue(callable(am.speak))

    def test_audio_manager_speak_delegates(self):
        """AudioManager.speak() must delegate to text_to_speech.speak()."""
        from app.audio import AudioManager

        with patch("app.audio.audio_manager.speak") as mock_speak:
            mock_speak.return_value = MagicMock(success=True)
            am = AudioManager()
            am.speak("Test delegation.")
            mock_speak.assert_called_once()

    def test_audio_manager_transcribe_delegates(self):
        """AudioManager.transcribe() must delegate to speech_to_text.transcribe()."""
        from app.audio import AudioManager

        with patch("app.audio.audio_manager.transcribe") as mock_transcribe:
            mock_transcribe.return_value = MagicMock(success=False, limitation="No backend.")
            am = AudioManager()
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _make_minimal_wav(wav_path)
            try:
                am.transcribe(wav_path)
                mock_transcribe.assert_called_once()
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)


# ---------------------------------------------------------------------------
# 9. Security / Privacy Assertions
# ---------------------------------------------------------------------------

class TestSecurityAndPrivacy(unittest.TestCase):

    def test_no_shell_true_in_vision(self):
        """Vision modules must not use shell=True."""
        import inspect
        import app.vision.screen as screen_mod
        import app.vision.camera as camera_mod
        import app.vision.image as image_mod

        for mod in [screen_mod, camera_mod, image_mod]:
            src = inspect.getsource(mod)
            self.assertNotIn(
                "shell=True",
                src,
                f"{mod.__name__} must not use shell=True.",
            )

    def test_no_shell_true_in_ocr(self):
        """OCR module must not use shell=True."""
        import inspect
        import app.models.ocr as ocr_mod

        src = inspect.getsource(ocr_mod)
        self.assertNotIn("shell=True", src)

    def test_no_continuous_screen_loop(self):
        """screen.py must not contain an infinite monitoring loop."""
        import inspect
        import app.vision.screen as screen_mod

        src = inspect.getsource(screen_mod)
        self.assertNotIn("while True", src)

    def test_no_independent_llama_subprocess_in_image(self):
        """image.py must not directly invoke llama-cli as a subprocess."""
        import inspect
        import app.vision.image as image_mod

        src = inspect.getsource(image_mod)
        # Must not hardcode subprocess calls to llama binaries
        self.assertNotIn("llama-cli", src)
        self.assertNotIn("llama-mtmd-cli", src)

    def test_no_independent_llama_subprocess_in_ocr(self):
        """ocr.py must not directly invoke llama-cli as a subprocess."""
        import inspect
        import app.models.ocr as ocr_mod

        src = inspect.getsource(ocr_mod)
        self.assertNotIn("llama-cli", src)

    def test_structured_image_result_fields(self):
        """ImageResult must contain all required fields."""
        from app.vision.image import ImageResult

        r = ImageResult(
            success=True,
            text="hello",
            model_id="qwen3-vl-4b",
            image_path="/tmp/img.bmp",
            elapsed_seconds=1.2,
            is_experimental=False,
        )
        self.assertTrue(r.success)
        self.assertEqual(r.text, "hello")
        self.assertFalse(r.is_experimental)

    def test_structured_speech_result_fields(self):
        """SpeechResult must contain all required fields."""
        from app.audio.speech_to_text import SpeechResult

        r = SpeechResult(
            success=False,
            backend=None,
            limitation="No backend available.",
            error="No local STT backend.",
        )
        self.assertFalse(r.success)
        self.assertIsNotNone(r.limitation)

    def test_structured_synthesis_result_fields(self):
        """SpeechSynthesisResult must contain all required fields."""
        from app.audio.text_to_speech import SpeechSynthesisResult

        r = SpeechSynthesisResult(
            success=True,
            text="Hello WHIS",
            backend="windows_sapi",
        )
        self.assertTrue(r.success)
        self.assertEqual(r.backend, "windows_sapi")


if __name__ == "__main__":
    unittest.main()
