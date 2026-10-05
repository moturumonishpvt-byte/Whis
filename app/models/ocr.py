"""OCR engine for WHIS.

Provides text extraction from images using local llama.cpp-based OCR models.

Primary:  GLM-OCR   (glm-ocr)     — verified
Fallback: DeepSeek-OCR (deepseek-ocr) — experimental

Usage:
    from app.models.ocr import OCREngine

    ocr = OCREngine()
    result = ocr.extract_text("document.bmp")           # auto: GLM → DeepSeek
    result = ocr.extract_text("doc.bmp", engine="glm") # GLM only
    result = ocr.extract_text("doc.bmp", engine="deepseek")

All inference goes through RuntimeAdapter. No independent subprocesses.
ONE-ACTIVE-MODEL rule must be respected by the caller.

DeepSeek-OCR is explicitly marked experimental; the caller is informed
via ImageResult.is_experimental = True.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.vision.image import ImageResult, validate_image, ImageValidationError

logger = logging.getLogger(__name__)

# Model IDs as registered in models.yaml
GLM_OCR_MODEL_ID = "glm-ocr"
DEEPSEEK_OCR_MODEL_ID = "deepseek-ocr"

# Standard OCR prompt — instructs the model to transcribe visible text
OCR_PROMPT = (
    "Please extract and transcribe all text visible in this image exactly as it appears. "
    "Output only the raw transcribed text without any commentary or formatting."
)


class OCRError(Exception):
    """Base exception for OCR engine failures."""


class OCREngine:
    """Local OCR engine for WHIS.

    Selects and invokes GLM-OCR (primary) or DeepSeek-OCR (fallback)
    through the existing RuntimeAdapter layer.
    """

    def __init__(
        self,
        runtime_adapter=None,
        model_registry=None,
    ) -> None:
        self._runtime_adapter = runtime_adapter
        self._model_registry = model_registry

    def _get_runtime_and_registry(self):
        """Lazy-initialize runtime and registry if not injected."""
        from app.ai.model_manager import ModelRegistry
        from app.ai.runtime import LlamaCppRuntime

        registry = self._model_registry or ModelRegistry()
        runtime = self._runtime_adapter or LlamaCppRuntime()
        return runtime, registry

    def _run_ocr_model(
        self,
        model_id: str,
        image_path: str,
        resolved_path,
        generation_config=None,
    ) -> ImageResult:
        """Run OCR inference for a single model.

        Args:
            model_id: Model ID to use.
            image_path: Original image path string.
            resolved_path: Validated, resolved Path object.
            generation_config: Optional GenerationConfig.

        Returns:
            ImageResult with extracted text or error.
        """
        try:
            from app.ai.runtime import GenerationConfig

            runtime, registry = self._get_runtime_and_registry()
            model_def = registry.get(model_id)

            is_experimental = model_def.is_experimental
            cfg = generation_config or GenerationConfig(max_new_tokens=1024, temperature=0.0)

            # Prompt for OCR model
            full_prompt = f"<image>\n{OCR_PROMPT}"

            t_start = time.monotonic()
            result = runtime.generate(
                model=model_def,
                prompt=full_prompt,
                config=cfg,
            )
            elapsed = time.monotonic() - t_start

            if result.success and result.text:
                return ImageResult(
                    success=True,
                    text=result.text,
                    model_id=model_id,
                    image_path=str(resolved_path),
                    elapsed_seconds=elapsed,
                    is_experimental=is_experimental,
                )
            else:
                return ImageResult(
                    success=False,
                    model_id=model_id,
                    error=result.error or "OCR inference returned no text.",
                    elapsed_seconds=elapsed,
                    is_experimental=is_experimental,
                )

        except Exception as exc:
            logger.error("OCR inference failed for model '%s': %s", model_id, exc)
            return ImageResult(
                success=False,
                model_id=model_id,
                error=str(exc),
            )

    def extract_text(
        self,
        image_path: str,
        engine: str = "auto",
        generation_config=None,
    ) -> ImageResult:
        """Extract text from an image using a local OCR model.

        Args:
            image_path: Path to the image file.
            engine: OCR engine selection:
                - "auto"     : GLM-OCR primary; DeepSeek-OCR fallback if GLM fails.
                - "glm"      : Force GLM-OCR only.
                - "deepseek" : Force DeepSeek-OCR only (experimental).
            generation_config: Optional GenerationConfig.

        Returns:
            ImageResult with extracted text, model used, and any error information.
            When DeepSeek-OCR is used, is_experimental=True is set on the result.
        """
        # Validate image
        try:
            resolved = validate_image(image_path)
        except ImageValidationError as exc:
            return ImageResult(success=False, error=str(exc))

        engine = engine.lower().strip()

        if engine == "glm":
            logger.info("OCR: using GLM-OCR (forced).")
            return self._run_ocr_model(GLM_OCR_MODEL_ID, image_path, resolved, generation_config=generation_config)

        if engine == "deepseek":
            logger.info("OCR: using DeepSeek-OCR (forced, experimental).")
            return self._run_ocr_model(DEEPSEEK_OCR_MODEL_ID, image_path, resolved, generation_config=generation_config)

        if engine == "auto":
            # Try GLM-OCR first
            logger.info("OCR auto-select: trying GLM-OCR (primary).")
            glm_result = self._run_ocr_model(GLM_OCR_MODEL_ID, image_path, resolved, generation_config=generation_config)

            if glm_result.success:
                logger.info("OCR: GLM-OCR succeeded.")
                return glm_result

            # Fall back to DeepSeek-OCR
            logger.warning(
                "OCR: GLM-OCR failed (%s). Falling back to DeepSeek-OCR (experimental).",
                glm_result.error,
            )
            deepseek_result = self._run_ocr_model(
                DEEPSEEK_OCR_MODEL_ID, image_path, resolved, generation_config=generation_config
            )

            if deepseek_result.success:
                logger.info("OCR: DeepSeek-OCR fallback succeeded (experimental).")
            else:
                logger.error(
                    "OCR: Both GLM-OCR and DeepSeek-OCR failed. "
                    "GLM error: %s | DeepSeek error: %s",
                    glm_result.error,
                    deepseek_result.error,
                )

            return deepseek_result

        # Unknown engine value
        return ImageResult(
            success=False,
            error=(
                f"Unknown OCR engine '{engine}'. "
                "Valid values: 'auto', 'glm', 'deepseek'."
            ),
        )
