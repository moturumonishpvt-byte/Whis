"""VisionEngine — unified API for screen capture, camera capture, and image understanding.

Usage:
    from app.vision import VisionEngine

    engine = VisionEngine()

    # Capture full screen
    result = engine.capture_screen()

    # Capture a region
    result = engine.capture_region(left=0, top=0, width=800, height=600)

    # Single camera frame (explicit call only)
    result = engine.capture_camera_frame()

    # Image understanding via Qwen3-VL-4B
    result = engine.understand_image("screen.bmp", "What is on the screen?")

    # OCR (delegates to OCREngine)
    result = engine.extract_text("document.bmp")

All calls are EXPLICIT. There is NO continuous screen monitoring, NO
always-on camera, NO background capture.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.vision.screen import CaptureResult, capture_region, capture_screen
from app.vision.camera import capture_camera_frame
from app.vision.image import ImageResult, understand_image, validate_image

logger = logging.getLogger(__name__)


class VisionEngine:
    """Unified API for WHIS vision capabilities.

    Wraps screen capture, camera capture, and image understanding.
    All operations are explicit on-demand calls.
    """

    def capture_screen(self, output_path: Optional[str] = None) -> CaptureResult:
        """Capture the full primary monitor screen.

        Args:
            output_path: File path for the captured image (BMP).
                         Defaults to a temp file.

        Returns:
            CaptureResult with success status, image path, and dimensions.
        """
        logger.debug("VisionEngine: explicit screen capture requested.")
        return capture_screen(output_path=output_path)

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        output_path: Optional[str] = None,
    ) -> CaptureResult:
        """Capture a rectangular region of the screen.

        Args:
            left: Left edge in screen coordinates.
            top: Top edge in screen coordinates.
            width: Width of region in pixels.
            height: Height of region in pixels.
            output_path: File path for the image. Defaults to a temp file.

        Returns:
            CaptureResult with success status, image path, and dimensions.
        """
        logger.debug(
            "VisionEngine: explicit region capture (%d,%d,%d,%d).", left, top, width, height
        )
        return capture_region(left, top, width, height, output_path=output_path)

    def capture_camera_frame(
        self,
        device_index: int = 0,
        output_path: Optional[str] = None,
    ) -> CaptureResult:
        """Capture a single frame from the camera (explicit call only).

        NO continuous camera monitoring. NO background camera access.

        Args:
            device_index: Camera device index (0 = default).
            output_path: File path for the frame. Defaults to a temp file.

        Returns:
            CaptureResult with success status, image path, and dimensions.
        """
        logger.debug("VisionEngine: explicit camera frame capture requested.")
        return capture_camera_frame(device_index=device_index, output_path=output_path)

    def understand_image(
        self,
        image_path: str,
        prompt: str,
        runtime_adapter=None,
        model_registry=None,
        generation_config=None,
    ) -> ImageResult:
        """Analyse an image using Qwen3-VL-4B.

        Inference goes through RuntimeAdapter exclusively (ONE-ACTIVE-MODEL
        rule must be satisfied by the caller before this call if a different
        model is currently loaded).

        Args:
            image_path: Path to the image file to analyse.
            prompt: Natural language instruction or question.
            runtime_adapter: Optional LlamaCppRuntime instance.
            model_registry: Optional ModelRegistry instance.
            generation_config: Optional GenerationConfig.

        Returns:
            ImageResult with model response or error details.
        """
        logger.debug("VisionEngine: understand_image called for '%s'.", image_path)
        return understand_image(
            image_path=image_path,
            prompt=prompt,
            runtime_adapter=runtime_adapter,
            model_registry=model_registry,
            generation_config=generation_config,
        )

    def extract_text(
        self,
        image_path: str,
        engine: str = "auto",
        runtime_adapter=None,
        model_registry=None,
    ) -> ImageResult:
        """Extract text from an image using OCR.

        Delegates to OCREngine. Engine selection:
          - "auto" : GLM-OCR primary, DeepSeek-OCR fallback
          - "glm"  : GLM-OCR only
          - "deepseek": DeepSeek-OCR only

        Args:
            image_path: Path to the image file.
            engine: OCR engine selection ("auto", "glm", "deepseek").
            runtime_adapter: Optional RuntimeAdapter instance.
            model_registry: Optional ModelRegistry instance.

        Returns:
            ImageResult with extracted text or error details.
        """
        from app.models.ocr import OCREngine

        ocr = OCREngine(
            runtime_adapter=runtime_adapter,
            model_registry=model_registry,
        )
        return ocr.extract_text(image_path=image_path, engine=engine)
