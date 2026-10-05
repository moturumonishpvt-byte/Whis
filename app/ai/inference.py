"""Inference engine for WHIS local model execution.

Provides a high-level ask/generate API that integrates context management,
model selection, model lifecycle management (via ModelLoader and UnloadManager),
and runtime execution (via RuntimeAdapter.generate()).

Architecture:
    InferenceEngine
        ↓
    ContextManager  (builds prompt, enforces context budget)
        ↓
    ModelLoader / UnloadManager  (lifecycle management)
        ↓
    RuntimeAdapter.generate()  (one-shot llama-cli call)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Dict, List, Optional

from app.ai.context import BuiltContext, ContextManager, ContextError, DEFAULT_GENERATION_TOKENS
from app.ai.lifecycle import ModelLifecycleState
from app.ai.loader import (
    ModelAlreadyLoadedError,
    ModelDisabledError,
    ModelLoadError,
    ModelLoader,
)
from app.ai.model_manager import ModelNotFoundError, ModelRegistry
from app.ai.ram_manager import InsufficientMemoryError, RAMManager
from app.ai.runtime import (
    GenerationConfig,
    InferenceResult,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeInferenceError,
)
from app.ai.unload import UnloadManager
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Deterministic model capability mapping (no LLM routing)
# ------------------------------------------------------------------

DEFAULT_MODEL_ID: str = "qwen3.5-4b"

CAPABILITY_MODEL_MAP: Dict[str, str] = {
    "chat": "qwen3.5-4b",
    "text_generation": "qwen3.5-4b",
    "reasoning": "lfm2.5-thinking-1.2b",
    "code": "qwen3-coder-30b",
    "coding": "qwen3-coder-30b",
    "vision": "qwen3-vl-4b",
    "ocr": "glm-ocr",
    "embedding": "qwen3-embedding-4b",
}


class InferenceError(WHISError):
    """Base exception for inference engine failures."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id


class ModelSelectionError(InferenceError):
    """Raised when no suitable model can be selected."""


@dataclass(frozen=True)
class InferenceResponse:
    """Full structured response from a single inference request."""

    text: str
    model_id: str
    success: bool
    elapsed_seconds: float
    load_seconds: float
    inference_seconds: float
    prompt_tokens_estimate: int = 0
    generated_tokens_estimate: int = 0
    model_reused: bool = False
    error: Optional[str] = None

    def __str__(self) -> str:
        return self.text


class InferenceEngine:
    """Orchestrates the full inference lifecycle for WHIS local model queries.

    Integrates ContextManager, ModelLoader, UnloadManager, and RuntimeAdapter
    to deliver a simple ask/generate interface without bypassing Stage 3
    lifecycle ownership.

    Usage::

        engine = InferenceEngine()
        response = engine.ask("What is binary search?")
        print(response.text)
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        runtime: Optional[RuntimeAdapter] = None,
        loader: Optional[ModelLoader] = None,
        unload_manager: Optional[UnloadManager] = None,
        ram_manager: Optional[RAMManager] = None,
        context_manager: Optional[ContextManager] = None,
        generation_tokens: int = DEFAULT_GENERATION_TOKENS,
    ) -> None:
        self.registry: ModelRegistry = registry if registry is not None else ModelRegistry()

        # Build runtime if not provided
        self.runtime: RuntimeAdapter = runtime if runtime is not None else LlamaCppRuntime()

        # Build ram manager if not provided
        self.ram_manager: Optional[RAMManager] = (
            ram_manager
            if ram_manager is not None
            else RAMManager(registry=self.registry)
        )

        # Build loader if not provided
        self.loader: ModelLoader = (
            loader
            if loader is not None
            else ModelLoader(
                registry=self.registry,
                runtime=self.runtime,
                ram_manager=self.ram_manager,
            )
        )

        # Build unload manager if not provided
        self.unload_manager: UnloadManager = (
            unload_manager
            if unload_manager is not None
            else UnloadManager(loader=self.loader, ram_manager=self.ram_manager)
        )

        # Context manager
        self.context_manager: ContextManager = (
            context_manager
            if context_manager is not None
            else ContextManager(generation_tokens=generation_tokens)
        )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_model_id(
        self,
        explicit_model_id: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> str:
        """Return the model ID to use for an inference request.

        Priority:
        1. explicit_model_id if provided.
        2. Capability mapping lookup if capability provided.
        3. DEFAULT_MODEL_ID fallback.
        """
        if explicit_model_id:
            return explicit_model_id
        if capability:
            return CAPABILITY_MODEL_MAP.get(capability.lower(), DEFAULT_MODEL_ID)
        return DEFAULT_MODEL_ID

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        capability: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        check_ram: bool = True,
        allow_warning: bool = True,
    ) -> InferenceResponse:
        """Run a full inference request through the local model subsystem.

        Steps:
        1. Select model (explicit → capability → default).
        2. Build and validate context prompt (ContextManager).
        3. Ensure model is loaded (reuse active or switch via UnloadManager).
        4. Run one-shot inference (RuntimeAdapter.generate).
        5. Return structured InferenceResponse.

        Args:
            prompt: User instruction or question.
            model_id: Explicit model override (skips capability mapping).
            system_prompt: Optional system instruction prepended to context.
            capability: Hint for capability-based model selection.
            config: GenerationConfig for max_new_tokens, temperature, etc.
            check_ram: Whether to enforce RAM safety check before loading.
            allow_warning: Whether to allow RAM WARNING state at load time.

        Returns:
            InferenceResponse with text, timing, and metadata.

        Raises:
            ContextError: If prompt is invalid or causes context overflow.
            InferenceError: On model selection, load, or runtime failure.
        """
        if not prompt or not prompt.strip():
            raise ContextError("prompt must not be empty")

        t_total_start = time.monotonic()

        # 1. Select target model
        target_id = self.select_model_id(model_id, capability)

        # Validate model exists before proceeding
        try:
            model_def = self.registry.get(target_id)
        except ModelNotFoundError as exc:
            raise InferenceError(
                f"Cannot run inference: model '{target_id}' is not registered.",
                model_id=target_id,
            ) from exc

        # 2. Build context prompt
        built_context = self.context_manager.build(
            user_prompt=prompt,
            model_id=target_id,
            model_context_tokens=model_def.default_context,
            system_prompt=system_prompt,
        )

        # 3. Ensure target model is loaded (reuse or switch)
        load_seconds, model_reused = self._ensure_model_loaded(
            target_id,
            check_ram=check_ram,
            allow_warning=allow_warning,
        )

        # 4. Run inference through RuntimeAdapter
        t_inf_start = time.monotonic()
        try:
            inference_result = self.runtime.generate(
                model=model_def,
                prompt=built_context.prompt,
                config=config,
            )
        except RuntimeInferenceError as exc:
            inference_seconds = time.monotonic() - t_inf_start
            total_seconds = time.monotonic() - t_total_start
            raise InferenceError(
                f"Inference failed for model '{target_id}': {exc}",
                model_id=target_id,
            ) from exc
        except RuntimeAdapterError as exc:
            raise InferenceError(
                f"Runtime error during inference for '{target_id}': {exc}",
                model_id=target_id,
            ) from exc

        inference_seconds = time.monotonic() - t_inf_start
        total_seconds = time.monotonic() - t_total_start

        logger.info(
            "InferenceEngine: completed request for model '%s' in %.2fs "
            "(load=%.2fs, inference=%.2fs).",
            target_id,
            total_seconds,
            load_seconds,
            inference_seconds,
        )

        return InferenceResponse(
            text=inference_result.text,
            model_id=target_id,
            success=True,
            elapsed_seconds=total_seconds,
            load_seconds=load_seconds,
            inference_seconds=inference_seconds,
            prompt_tokens_estimate=inference_result.prompt_tokens_estimate,
            generated_tokens_estimate=inference_result.generated_tokens_estimate,
            model_reused=model_reused,
        )

    def ask(
        self,
        question: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        capability: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> InferenceResponse:
        """Convenience wrapper around generate() for simple question-answer requests.

        Args:
            question: The user question or instruction.
            model_id: Explicit model override.
            system_prompt: Optional system instruction.
            capability: Hint for deterministic model selection.
            config: GenerationConfig override.

        Returns:
            InferenceResponse with text and metadata.
        """
        return self.generate(
            prompt=question,
            model_id=model_id,
            system_prompt=system_prompt,
            capability=capability,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal lifecycle helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(
        self,
        target_model_id: str,
        check_ram: bool = True,
        allow_warning: bool = True,
    ) -> tuple[float, bool]:
        """Ensure the target model is loaded and active.

        Returns:
            (load_seconds, model_reused) tuple.
        """
        t_load_start = time.monotonic()

        # Case 1: Target model already active → reuse
        if self.loader.is_loaded(target_model_id):
            load_seconds = time.monotonic() - t_load_start
            logger.debug("InferenceEngine: reusing active model '%s'.", target_model_id)
            return load_seconds, True

        # Case 2: Different model active → unload it first
        if self.loader.is_loaded():
            active_id = self.loader._loaded_model_id
            logger.info(
                "InferenceEngine: switching from '%s' to '%s'.",
                active_id,
                target_model_id,
            )
            result = self.unload_manager.unload_current(
                reason=f"Switching to model '{target_model_id}'"
            )
            if not result.success:
                raise InferenceError(
                    f"Cannot switch to model '{target_model_id}': "
                    f"failed to unload active model '{active_id}': {result.reason}",
                    model_id=target_model_id,
                )

        # Case 3: No model active → load target
        try:
            self.loader.load(
                target_model_id,
                check_ram=check_ram,
                allow_warning=allow_warning,
            )
        except ModelNotFoundError as exc:
            raise InferenceError(str(exc), model_id=target_model_id) from exc
        except ModelDisabledError as exc:
            raise InferenceError(str(exc), model_id=target_model_id) from exc
        except InsufficientMemoryError as exc:
            raise InferenceError(str(exc), model_id=target_model_id) from exc
        except ModelLoadError as exc:
            raise InferenceError(str(exc), model_id=target_model_id) from exc

        load_seconds = time.monotonic() - t_load_start
        return load_seconds, False

    def unload(self) -> None:
        """Unload any currently active model through UnloadManager."""
        if self.loader.is_loaded():
            self.unload_manager.unload_current(reason="InferenceEngine explicit unload")

    @property
    def active_model_id(self) -> Optional[str]:
        """Return the currently loaded model ID, or None."""
        return self.loader._loaded_model_id if self.loader.is_loaded() else None

    @property
    def is_model_loaded(self) -> bool:
        """Return True if any model is currently loaded and ready."""
        return self.loader.is_loaded()
