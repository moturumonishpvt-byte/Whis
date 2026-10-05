"""Unload Manager orchestration layer for WHIS.

Provides deterministic, memory-aware policies and controls for deciding
WHEN and HOW currently active models are unloaded through ModelLoader.
Enforces single-active-model concurrency, active model protection,
lifecycle consistency, and cache metadata preservation without directly
owning or manipulating llama.cpp subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

from app.ai.cache import CacheState, ModelCache, ModelCacheEntry
from app.ai.lifecycle import ModelLifecycleState
from app.ai.loader import ModelLoader, ModelLoaderError
from app.ai.model_manager import ModelNotFoundError
from app.ai.ram_manager import MemoryDecision, RAMManager, SafetyEvaluation
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)


class UnloadAction(str, Enum):
    """Action recommendation resulting from an unload evaluation."""

    NO_ACTION = "NO_ACTION"
    UNLOAD_REQUIRED = "UNLOAD_REQUIRED"
    UNLOAD_NOT_POSSIBLE = "UNLOAD_NOT_POSSIBLE"
    NO_ACTIVE_MODEL = "NO_ACTIVE_MODEL"


class ModelUnloadError(WHISError):
    """Base exception for model unloading operations."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id


@dataclass(frozen=True)
class UnloadDecision:
    """Structured decision explaining whether unloading is recommended or required."""

    action: UnloadAction
    model_id: Optional[str]
    reason: str
    memory_decision: Optional[MemoryDecision] = None
    evaluation: Optional[SafetyEvaluation] = None

    @property
    def should_unload(self) -> bool:
        """Return True if the decision requires unloading."""
        return self.action == UnloadAction.UNLOAD_REQUIRED


@dataclass(frozen=True)
class UnloadResult:
    """Structured outcome of an unload operation."""

    success: bool
    model_id: Optional[str]
    previous_state: Optional[ModelLifecycleState]
    final_state: Optional[ModelLifecycleState]
    reason: str
    error: Optional[str] = None


class UnloadManager:
    """Orchestrates deterministic model unloading through ModelLoader.

    Enforces active model protection, delegates process termination to ModelLoader,
    ensures cache metadata is preserved, and computes memory-aware unload decisions
    without spawning background threads, schedulers, or watchers.
    """

    def __init__(
        self,
        loader: ModelLoader,
        ram_manager: Optional[RAMManager] = None,
        cache: Optional[ModelCache] = None,
    ) -> None:
        self.loader: ModelLoader = loader
        self.ram_manager: Optional[RAMManager] = (
            ram_manager if ram_manager is not None else loader.ram_manager
        )
        self.cache: Optional[ModelCache] = (
            cache if cache is not None else loader.cache
        )

    @property
    def active_model_id(self) -> Optional[str]:
        """Return the model identifier of the currently active model, or None."""
        if not self.loader.is_loaded():
            return None
        return self.loader._loaded_model_id

    @property
    def has_active_model(self) -> bool:
        """Return True if a model is currently loaded and running."""
        return self.loader.is_loaded()

    @property
    def active_lifecycle_state(self) -> ModelLifecycleState:
        """Return the current lifecycle state of the loader's model."""
        return self.loader.lifecycle.state

    def can_unload(self, model_id: Optional[str] = None) -> bool:
        """Return whether a model can currently be safely unloaded.

        Deterministic criteria:
        - If model_id is specified: the model must be currently loaded and active.
        - If model_id is None: any model must be currently loaded and active.
        - Lifecycle state must permit unloading (READY or FAILED).
        - Transitional states (LOADING, STOPPING) and active execution (BUSY) cannot be unloaded.
        - If no model is active, returns False.
        """
        if not self.loader.is_loaded():
            return False

        if model_id is not None and self.loader._loaded_model_id != model_id:
            return False

        current_state = self.loader.lifecycle.state
        return current_state in {ModelLifecycleState.READY, ModelLifecycleState.FAILED}

    def unload_current(
        self,
        timeout: Optional[float] = None,
        reason: str = "Explicit unload",
    ) -> UnloadResult:
        """Unload the currently active model through ModelLoader.

        Ensures:
        - Lifecycle reaches UNLOADED.
        - Runtime reaches STOPPED.
        - Cache entry transitions to CACHED_METADATA and preserves metadata.
        - Failure information is captured if unloading encounters an error.
        """
        if not self.loader.is_loaded():
            return UnloadResult(
                success=False,
                model_id=None,
                previous_state=self.loader.lifecycle.state,
                final_state=self.loader.lifecycle.state,
                reason="No active model is currently loaded.",
            )

        active_id = self.loader._loaded_model_id
        prev_state = self.loader.lifecycle.state

        if not self.can_unload():
            return UnloadResult(
                success=False,
                model_id=active_id,
                previous_state=prev_state,
                final_state=self.loader.lifecycle.state,
                reason=f"Model '{active_id}' cannot be unloaded while in '{prev_state.value}' state.",
            )

        try:
            self.loader.unload(timeout=timeout)
            return UnloadResult(
                success=True,
                model_id=active_id,
                previous_state=prev_state,
                final_state=self.loader.lifecycle.state,
                reason=reason,
            )
        except Exception as exc:
            logger.error("Failed to unload model '%s': %s", active_id, exc)
            return UnloadResult(
                success=False,
                model_id=active_id,
                previous_state=prev_state,
                final_state=self.loader.lifecycle.state,
                reason=f"Failed to unload model '{active_id}': {exc}",
                error=str(exc),
            )

    def unload_model(
        self,
        model_id: str,
        timeout: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> UnloadResult:
        """Unload a specific model if it is currently active.

        Enforces active model protection: if the specified model is not active,
        does NOT touch the active model and returns a safe unsuccessful result.
        """
        if not self.loader.is_loaded():
            return UnloadResult(
                success=False,
                model_id=model_id,
                previous_state=self.loader.lifecycle.state,
                final_state=self.loader.lifecycle.state,
                reason=f"Cannot unload model '{model_id}': no model is currently loaded.",
            )

        if self.loader._loaded_model_id != model_id:
            return UnloadResult(
                success=False,
                model_id=model_id,
                previous_state=self.loader.lifecycle.state,
                final_state=self.loader.lifecycle.state,
                reason=(
                    f"Cannot unload model '{model_id}': active model is "
                    f"'{self.loader._loaded_model_id}'. Active model is protected."
                ),
            )

        return self.unload_current(
            timeout=timeout,
            reason=reason or f"Explicit unload of model '{model_id}'",
        )

    def get_unload_candidate(self, exclude_active: bool = True) -> Optional[ModelCacheEntry]:
        """Identify the least recently used model candidate using ModelCache LRU.

        Identifies candidate metadata only; does NOT automatically unload any model.
        When exclude_active=True (default), protects the currently active model
        from selection when looking for inactive cached candidates.
        """
        if self.cache is None:
            return None
        return self.cache.get_least_recently_used(exclude_active=exclude_active)

    def unload_if_needed(
        self,
        target_model_id: Optional[str] = None,
        context_tokens: Optional[int] = None,
        memory_threshold_bytes: Optional[int] = None,
    ) -> UnloadDecision:
        """Deterministic policy helper to evaluate whether unloading is needed.

        Does NOT automatically execute unloads. Evaluates:
        1. If no model is active -> NO_ACTIVE_MODEL.
        2. If target_model_id is specified:
           - Target model matches active model -> NO_ACTION.
           - Active model cannot be unloaded (e.g. busy) -> UNLOAD_NOT_POSSIBLE.
           - Target model is unregistered in ModelRegistry -> UNLOAD_NOT_POSSIBLE.
           - Target model cannot fit into total physical safe RAM even if active model is unloaded -> UNLOAD_NOT_POSSIBLE.
           - Otherwise -> UNLOAD_REQUIRED (enforces single active model concurrency).
        3. If target_model_id is None (RAM pressure check):
           - Available RAM < safety reserve (or memory_threshold_bytes) ->
             if can_unload(): UNLOAD_REQUIRED else UNLOAD_NOT_POSSIBLE.
           - System memory healthy -> NO_ACTION.
        """
        if not self.has_active_model:
            return UnloadDecision(
                action=UnloadAction.NO_ACTIVE_MODEL,
                model_id=None,
                reason="No active model is currently loaded.",
            )

        active_id = self.active_model_id

        # Target model evaluation
        if target_model_id is not None:
            if target_model_id == active_id:
                return UnloadDecision(
                    action=UnloadAction.NO_ACTION,
                    model_id=active_id,
                    reason=f"Model '{target_model_id}' is already loaded and active.",
                )

            if not self.can_unload():
                return UnloadDecision(
                    action=UnloadAction.UNLOAD_NOT_POSSIBLE,
                    model_id=active_id,
                    reason=(
                        f"Active model '{active_id}' cannot be unloaded while in "
                        f"'{self.loader.lifecycle.state.value}' state."
                    ),
                )

            if self.ram_manager is not None:
                try:
                    estimate = self.ram_manager.estimate_model_memory(
                        target_model_id,
                        context_tokens=context_tokens,
                    )
                except ModelNotFoundError:
                    return UnloadDecision(
                        action=UnloadAction.UNLOAD_NOT_POSSIBLE,
                        model_id=active_id,
                        reason=f"Target model '{target_model_id}' is not registered in ModelRegistry.",
                    )

                snapshot = self.ram_manager.get_system_memory()
                max_safe_ram = max(0, snapshot.total_bytes - self.ram_manager.safety_reserve_bytes)
                if estimate.estimated_total_bytes > max_safe_ram:
                    return UnloadDecision(
                        action=UnloadAction.UNLOAD_NOT_POSSIBLE,
                        model_id=active_id,
                        reason=(
                            f"Target model '{target_model_id}' estimated requirement "
                            f"({estimate.estimated_total_gb} GB) exceeds total system safe RAM "
                            f"({round(max_safe_ram / (1024**3), 2)} GB) even if active model is unloaded."
                        ),
                        memory_decision=MemoryDecision.UNSAFE,
                    )

                safety = self.ram_manager.evaluate_safety(
                    target_model_id,
                    context_tokens=context_tokens,
                )
                return UnloadDecision(
                    action=UnloadAction.UNLOAD_REQUIRED,
                    model_id=active_id,
                    reason=(
                        f"Unloading active model '{active_id}' is required to load "
                        f"'{target_model_id}' (enforces single active model concurrency)."
                    ),
                    memory_decision=safety.decision,
                    evaluation=safety,
                )

            return UnloadDecision(
                action=UnloadAction.UNLOAD_REQUIRED,
                model_id=active_id,
                reason=(
                    f"Unloading active model '{active_id}' is required to load "
                    f"'{target_model_id}' (enforces single active model concurrency)."
                ),
            )

        # System RAM pressure evaluation (target_model_id is None)
        if self.ram_manager is not None:
            snapshot = self.ram_manager.get_system_memory()
            reserve = (
                memory_threshold_bytes
                if memory_threshold_bytes is not None
                else self.ram_manager.safety_reserve_bytes
            )

            if snapshot.available_bytes < reserve:
                if not self.can_unload():
                    return UnloadDecision(
                        action=UnloadAction.UNLOAD_NOT_POSSIBLE,
                        model_id=active_id,
                        reason=(
                            f"System RAM pressure detected ({snapshot.available_gb} GB available < "
                            f"{round(reserve / (1024**3), 2)} GB reserve), but active model cannot "
                            f"be unloaded in '{self.loader.lifecycle.state.value}' state."
                        ),
                        memory_decision=MemoryDecision.UNSAFE,
                    )
                return UnloadDecision(
                    action=UnloadAction.UNLOAD_REQUIRED,
                    model_id=active_id,
                    reason=(
                        f"System RAM pressure detected ({snapshot.available_gb} GB available < "
                        f"{round(reserve / (1024**3), 2)} GB reserve). Unload recommended to "
                        f"restore memory safety."
                    ),
                    memory_decision=MemoryDecision.UNSAFE,
                )

            return UnloadDecision(
                action=UnloadAction.NO_ACTION,
                model_id=active_id,
                reason=(
                    f"System memory is healthy ({snapshot.available_gb} GB available >= "
                    f"{round(reserve / (1024**3), 2)} GB reserve). No unload needed."
                ),
                memory_decision=MemoryDecision.SAFE,
            )

        return UnloadDecision(
            action=UnloadAction.NO_ACTION,
            model_id=active_id,
            reason="No memory manager configured and no target model specified. No action needed.",
        )
