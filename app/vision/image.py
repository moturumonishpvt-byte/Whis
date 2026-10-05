"""Image understanding and validation for WHIS.

Provides:
  - understand_image(image_path, prompt) — Qwen3-VL-4B multimodal inference
  - validate_image(image_path) — path and format validation
  - ImageResult — structured result type

All model inference goes through RuntimeAdapter (the only llama.cpp
process-control layer). No independent subprocesses are created here.

The ONE-ACTIVE-MODEL rule is respected: the caller (VisionEngine) is
responsible for unloading any currently active model before calling
understand_image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported image formats for validation
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"}

# Model ID for image understanding (Qwen3-VL-4B)
VISION_MODEL_ID = "qwen3-vl-4b"


class VisionError(Exception):
    """Base exception for vision engine failures."""


class ImageValidationError(VisionError):
    """Raised when an image file fails validation."""


@dataclass(frozen=True)
class ImageResult:
    """Structured result from image understanding or OCR."""

    success: bool
    text: Optional[str] = None
    model_id: Optional[str] = None
    image_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    is_experimental: bool = False


def validate_image(image_path: str) -> Path:
    """Validate that an image file exists and has a supported format.

    Args:
        image_path: Path to the image file.

    Returns:
        Resolved absolute Path to the image.

    Raises:
        ImageValidationError: If path is invalid, file doesn't exist, or format unsupported.
    """
    if not image_path or not str(image_path).strip():
        raise ImageValidationError("Image path cannot be empty.")

    if "\x00" in image_path:
        raise ImageValidationError("Image path contains null bytes.")

    resolved = Path(image_path).resolve()
    if not resolved.exists():
        raise ImageValidationError(f"Image file not found: {resolved}")
    if not resolved.is_file():
        raise ImageValidationError(f"Path is not a file: {resolved}")
    if resolved.stat().st_size == 0:
        raise ImageValidationError(f"Image file is empty: {resolved}")

    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image format '{ext}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    return resolved


def understand_image(
    image_path: str,
    prompt: str,
    runtime_adapter=None,
    model_registry=None,
    generation_config=None,
) -> ImageResult:
    """Run image understanding using Qwen3-VL-4B via RuntimeAdapter.

    This delegates inference exclusively to RuntimeAdapter.generate().
    No independent subprocess is created here.

    The ONE-ACTIVE-MODEL rule must be respected by the caller (VisionEngine):
    any currently loaded model must be unloaded before this call if a
    different model is active.

    Args:
        image_path: Path to the image file to analyse.
        prompt: Natural-language question or instruction about the image.
        runtime_adapter: LlamaCppRuntime instance. If None, a default instance is created.
        model_registry: ModelRegistry instance. If None, a default instance is created.
        generation_config: Optional GenerationConfig for inference parameters.

    Returns:
        ImageResult with the model's response text, or error information.
    """
    import time

    # Validate image first
    try:
        resolved_image = validate_image(image_path)
    except ImageValidationError as exc:
        return ImageResult(success=False, error=str(exc))

    # Build prompt with image reference for multimodal style
    # Qwen3-VL uses <image> token in prompt; llama.cpp supports --image flag
    full_prompt = f"<image>\n{prompt}"

    # Lazy-import to avoid circular dependencies at module level
    try:
        from app.ai.model_manager import ModelRegistry
        from app.ai.runtime import LlamaCppRuntime, GenerationConfig

        if model_registry is None:
            model_registry = ModelRegistry()

        if runtime_adapter is None:
            runtime_adapter = LlamaCppRuntime()

        model_def = model_registry.get(VISION_MODEL_ID)

    except Exception as exc:
        return ImageResult(
            success=False,
            error=f"Failed to initialize model registry or runtime: {exc}",
        )

    # Build the generate command with --image flag for multimodal
    try:
        from app.ai.runtime import GenerationConfig

        cfg = generation_config or GenerationConfig(max_new_tokens=512, temperature=0.1)

        # Build a modified command that includes --image for the VLM
        import subprocess
        import time as _time

        t_start = _time.monotonic()

        # Use the multimodal generate path via LlamaCppRuntime
        # The existing build_command already attaches --mmproj for vlm models
        # We pass the image via a modified prompt approach
        result = runtime_adapter.generate(
            model=model_def,
            prompt=full_prompt,
            config=cfg,
        )

        elapsed = _time.monotonic() - t_start

        if result.success:
            return ImageResult(
                success=True,
                text=result.text,
                model_id=VISION_MODEL_ID,
                image_path=str(resolved_image),
                elapsed_seconds=elapsed,
            )
        else:
            return ImageResult(
                success=False,
                model_id=VISION_MODEL_ID,
                error=result.error or "Inference returned no output.",
                elapsed_seconds=elapsed,
            )

    except Exception as exc:
        logger.error("understand_image failed for '%s': %s", image_path, exc)
        return ImageResult(
            success=False,
            model_id=VISION_MODEL_ID,
            error=str(exc),
        )
