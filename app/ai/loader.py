"""Model loader orchestration layer.

Responsible for orchestrating local model loading and unloading through the
RuntimeAdapter, enforcing single-model execution, verifying model prerequisites,
and tracking active model state without containing llama.cpp subprocess details.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional, Union

from app.ai.lifecycle import (
    InvalidStateTransitionError,
    ModelLifecycle,
    ModelLifecycleError,
    ModelLifecycleState,
    ModelRuntimeStatus,
)
from app.ai.model_manager import (
    ModelDefinition,
    ModelNotFoundError,
    ModelRegistry,
)
from app.ai.runtime import (
    DevicePolicy,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeState,
)
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)


class ModelLoaderError(WHISError):
    """Base exception for model loader operations."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id

    def __str__(self) -> str:
        base = super().__str__()
        if self.model_id:
            return f"{base} (model_id='{self.model_id}')"
        return base


class ModelDisabledError(ModelLoaderError):
    """Raised when an attempt is made to load a disabled model."""


class ModelAlreadyLoadedError(ModelLoaderError):
    """Raised when attempting to load a model while another is already active."""


class ModelLoadError(ModelLoaderError):
    """Raised when model verification or runtime initialization fails."""


@dataclass(frozen=True)
class LoadedModelInfo:
    """Structured status information about the active model and its runtime."""

    model_id: str
    display_name: str
    type: str
    role: str
    status: str
    runtime_state: RuntimeState
    model_path: str
    mmproj_path: Optional[str]
    default_context: int
    device_policy: DevicePolicy
    executable: str
    loaded_at: float
    definition: ModelDefinition
    lifecycle_state: ModelLifecycleState = ModelLifecycleState.READY


class ModelLoader:
    """Orchestrates model lifecycle by bridging ModelRegistry and RuntimeAdapter.

    Enforces single-active-model policy suitable for Intel Core i3 / 16 GB RAM baseline,
    verifies model status and file prerequisites, and exposes structured status.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        runtime: Optional[RuntimeAdapter] = None,
        lifecycle: Optional[ModelLifecycle] = None,
    ) -> None:
        self.registry: ModelRegistry = registry if registry is not None else ModelRegistry()
        self.runtime: RuntimeAdapter = runtime if runtime is not None else LlamaCppRuntime()
        self.lifecycle: ModelLifecycle = lifecycle if lifecycle is not None else ModelLifecycle()
        self._loaded_model_id: Optional[str] = None
        self._loaded_model_def: Optional[ModelDefinition] = None
        self._loaded_at: Optional[float] = None

    def _cleanup_internal_state(self) -> None:
        """Reset internal tracking state."""
        self._loaded_model_id = None
        self._loaded_model_def = None
        self._loaded_at = None

    def is_loaded(self, model_id: Optional[str] = None) -> bool:
        """Check whether a model (or a specific model ID) is currently loaded and running.

        If the underlying process has exited in the background, automatically synchronizes
        internal state and returns False.
        """
        self.lifecycle.sync_with_runtime(self.runtime)

        if not self.runtime.is_running():
            self._cleanup_internal_state()
            return False

        if model_id is not None:
            return self._loaded_model_id == model_id

        return self._loaded_model_id is not None

    def loaded_model(self) -> Optional[ModelDefinition]:
        """Return the active ModelDefinition, or None if no model is loaded."""
        if not self.is_loaded():
            return None
        return self._loaded_model_def

    def get_status(self) -> Optional[LoadedModelInfo]:
        """Return structured runtime and model status, or None if no model is loaded."""
        if not self.is_loaded() or self._loaded_model_def is None or self._loaded_at is None:
            return None

        # Extract device policy and executable from runtime config if available
        config = getattr(self.runtime, "config", None)
        device_policy = getattr(config, "device_policy", DevicePolicy.CPU_ONLY)
        executable = getattr(config, "executable", "llama-cli")

        return LoadedModelInfo(
            model_id=self._loaded_model_def.id,
            display_name=self._loaded_model_def.display_name,
            type=self._loaded_model_def.type,
            role=self._loaded_model_def.role,
            status=self._loaded_model_def.status,
            runtime_state=self.runtime.state,
            model_path=self._loaded_model_def.model_path,
            mmproj_path=self._loaded_model_def.mmproj_path,
            default_context=self._loaded_model_def.default_context,
            device_policy=device_policy,
            executable=executable,
            loaded_at=self._loaded_at,
            definition=self._loaded_model_def,
            lifecycle_state=self.lifecycle.state,
        )

    def get_lifecycle_status(self) -> ModelRuntimeStatus:
        """Return high-level ModelRuntimeStatus from the lifecycle state machine."""
        return self.lifecycle.get_status(self.runtime)

    def load(self, model_id: str) -> LoadedModelInfo:
        """Load a registered model through the RuntimeAdapter.

        Enforces single-model concurrency. Validates model registration, enabled flag,
        and existence of required weight and projector files before invoking runtime.

        Args:
            model_id: Identifier of model registered in ModelRegistry.

        Returns:
            LoadedModelInfo describing the active model and runtime.

        Raises:
            ModelAlreadyLoadedError: If another model (or this model) is already active.
            ModelNotFoundError: If model_id is not registered.
            ModelDisabledError: If the model is marked enabled=False.
            ModelLoadError: If required files are missing or runtime startup fails.
        """
        # 1. Check single active model rule
        if self.is_loaded():
            if self._loaded_model_id == model_id:
                raise ModelAlreadyLoadedError(
                    f"Model '{model_id}' is already loaded and running.",
                    model_id=model_id,
                )
            raise ModelAlreadyLoadedError(
                f"Cannot load model '{model_id}': model '{self._loaded_model_id}' is already active. Call unload() first.",
                model_id=model_id,
            )

        # 2. Lookup model in registry (raises ModelNotFoundError if unknown)
        model = self.registry.get(model_id)

        # 3. Check enabled flag
        if not model.enabled:
            raise ModelDisabledError(
                f"Model '{model_id}' is disabled in configuration.",
                model_id=model_id,
            )

        # 4. Validate physical files on disk
        model_path = self.registry.resolve_model_path(model_id)
        if not model_path.is_file():
            raise ModelLoadError(
                f"Model weights file not found on disk at: {model_path}",
                model_id=model_id,
            )

        if model.has_projector and model.mmproj_path:
            mmproj_path = self.registry.resolve_mmproj_path(model_id)
            if mmproj_path is None or not mmproj_path.is_file():
                raise ModelLoadError(
                    f"Multimodal projector file not found on disk at: {mmproj_path}",
                    model_id=model_id,
                )

        # 5. Transition lifecycle: UNLOADED -> LOADING
        self.lifecycle.begin_loading(model_id)

        # 6. Delegate startup to RuntimeAdapter
        try:
            self.runtime.start(model)
        except RuntimeAdapterError as exc:
            self.lifecycle.mark_failed(str(exc))
            self._cleanup_internal_state()
            if self.runtime.is_running():
                try:
                    self.runtime.stop()
                except Exception:
                    pass
            raise ModelLoadError(
                f"Failed to start runtime for model '{model_id}': {exc}",
                model_id=model_id,
            ) from exc
        except Exception as exc:
            self.lifecycle.mark_failed(str(exc))
            self._cleanup_internal_state()
            if self.runtime.is_running():
                try:
                    self.runtime.stop()
                except Exception:
                    pass
            raise ModelLoadError(
                f"Unexpected failure while launching runtime for model '{model_id}': {exc}",
                model_id=model_id,
            ) from exc

        # 7. Transition lifecycle: LOADING -> READY
        self.lifecycle.mark_ready()

        # 8. Record active model state
        self._loaded_model_id = model_id
        self._loaded_model_def = model
        self._loaded_at = time.time()

        status = self.get_status()
        if status is None:
            self.lifecycle.mark_failed("Runtime process died immediately after launch")
            self._cleanup_internal_state()
            raise ModelLoadError(
                f"Runtime process for model '{model_id}' died immediately after launch.",
                model_id=model_id,
            )

        logger.info("ModelLoader successfully loaded model '%s' (State: READY).", model_id)
        return status

    def unload(self, timeout: Optional[float] = None) -> None:
        """Unload the active model through RuntimeAdapter and reset tracking state."""
        try:
            if self.lifecycle.state not in {ModelLifecycleState.UNLOADED, ModelLifecycleState.FAILED}:
                self.lifecycle.begin_stopping()
            self.runtime.stop(timeout=timeout)
        finally:
            self.lifecycle.mark_unloaded()
            self._cleanup_internal_state()
            logger.info("ModelLoader unloaded active model (State: UNLOADED).")
