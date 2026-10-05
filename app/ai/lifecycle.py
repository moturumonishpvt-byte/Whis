"""Model lifecycle and state management layer.

Maintains authoritative high-level lifecycle state for local models, validates
explicit state transitions, records timezone-aware UTC timestamps, and synchronizes
with the underlying RuntimeAdapter without managing OS processes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import List, Optional, Set

from app.ai.runtime import RuntimeAdapter, RuntimeState
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)


class ModelLifecycleState(str, Enum):
    """High-level lifecycle state of a WHIS model."""

    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    BUSY = "BUSY"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


class ModelLifecycleError(WHISError):
    """Base exception for model lifecycle errors."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id


class InvalidStateTransitionError(ModelLifecycleError):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        current_state: ModelLifecycleState,
        attempted_state: ModelLifecycleState,
        model_id: Optional[str] = None,
        expected_states: Optional[List[ModelLifecycleState]] = None,
    ) -> None:
        expected_str = (
            f" Expected state: {', '.join(s.value for s in expected_states)}."
            if expected_states
            else ""
        )
        model_str = f" for model '{model_id}'" if model_id else ""
        message = (
            f"Cannot transition model{model_str} from {current_state.value} to "
            f"{attempted_state.value}.{expected_str}"
        )
        super().__init__(message, model_id=model_id)
        self.current_state = current_state
        self.attempted_state = attempted_state
        self.expected_states = expected_states or []


@dataclass(frozen=True)
class ModelRuntimeStatus:
    """Structured snapshot of model lifecycle and runtime state."""

    model_id: Optional[str]
    lifecycle_state: ModelLifecycleState
    runtime_state: RuntimeState
    loaded_at: Optional[datetime]
    ready_at: Optional[datetime]
    last_used_at: Optional[datetime]
    stopped_at: Optional[datetime]
    failure_reason: Optional[str]
    operation_count: int
    error_count: int


# Mapping of valid transitions: current_state -> set of allowable next states
VALID_TRANSITIONS: dict[ModelLifecycleState, Set[ModelLifecycleState]] = {
    ModelLifecycleState.UNLOADED: {
        ModelLifecycleState.LOADING,
    },
    ModelLifecycleState.LOADING: {
        ModelLifecycleState.READY,
        ModelLifecycleState.FAILED,
        ModelLifecycleState.STOPPING,
    },
    ModelLifecycleState.READY: {
        ModelLifecycleState.BUSY,
        ModelLifecycleState.STOPPING,
        ModelLifecycleState.FAILED,
    },
    ModelLifecycleState.BUSY: {
        ModelLifecycleState.READY,
        ModelLifecycleState.STOPPING,
        ModelLifecycleState.FAILED,
    },
    ModelLifecycleState.STOPPING: {
        ModelLifecycleState.UNLOADED,
        ModelLifecycleState.FAILED,
    },
    ModelLifecycleState.FAILED: {
        ModelLifecycleState.UNLOADED,
        ModelLifecycleState.STOPPING,
        ModelLifecycleState.LOADING,
    },

}


class ModelLifecycle:
    """Explicit state machine governing WHIS model lifecycle.

    Maintains lifecycle status, validates legal state transitions, records
    UTC timestamps, and synchronizes with the process-level RuntimeAdapter.
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self._state: ModelLifecycleState = ModelLifecycleState.UNLOADED
        self._model_id: Optional[str] = model_id
        self._loaded_at: Optional[datetime] = None
        self._ready_at: Optional[datetime] = None
        self._last_used_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
        self._failure_reason: Optional[str] = None
        self._operation_count: int = 0
        self._error_count: int = 0

    @property
    def state(self) -> ModelLifecycleState:
        """Return current lifecycle state."""
        return self._state

    @property
    def model_id(self) -> Optional[str]:
        """Return identifier of the active model, if any."""
        return self._model_id

    @property
    def loaded_at(self) -> Optional[datetime]:
        """Return timestamp when model loading began."""
        return self._loaded_at

    @property
    def ready_at(self) -> Optional[datetime]:
        """Return timestamp when model became ready for operations."""
        return self._ready_at

    @property
    def last_used_at(self) -> Optional[datetime]:
        """Return timestamp when model was last used."""
        return self._last_used_at

    @property
    def stopped_at(self) -> Optional[datetime]:
        """Return timestamp when model was unloaded or stopped."""
        return self._stopped_at

    @property
    def failure_reason(self) -> Optional[str]:
        """Return reason string if model failed."""
        return self._failure_reason

    @property
    def operation_count(self) -> int:
        """Return total number of operations initiated on this model."""
        return self._operation_count

    @property
    def error_count(self) -> int:
        """Return total number of errors encountered during this lifecycle."""
        return self._error_count

    def is_ready(self) -> bool:
        """Check whether the model is in READY state."""
        return self._state == ModelLifecycleState.READY

    def is_busy(self) -> bool:
        """Check whether the model is in BUSY state."""
        return self._state == ModelLifecycleState.BUSY

    def is_loaded(self) -> bool:
        """Check whether the model is loaded and usable (READY or BUSY)."""
        return self._state in {ModelLifecycleState.READY, ModelLifecycleState.BUSY}

    def _transition(
        self,
        new_state: ModelLifecycleState,
        expected_states: Optional[List[ModelLifecycleState]] = None,
    ) -> None:
        """Enforce transition rules and transition to new_state."""
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                current_state=self._state,
                attempted_state=new_state,
                model_id=self._model_id,
                expected_states=expected_states or sorted(allowed, key=lambda s: s.value),
            )
        old_state = self._state
        self._state = new_state
        logger.debug(
            "ModelLifecycle [%s]: transitioned %s -> %s",
            self._model_id or "none",
            old_state.value,
            new_state.value,
        )

    def begin_loading(self, model_id: str) -> None:
        """Begin loading a model. Transitions UNLOADED or FAILED -> LOADING."""
        self._transition(
            ModelLifecycleState.LOADING,
            expected_states=[ModelLifecycleState.UNLOADED, ModelLifecycleState.FAILED],
        )

        self._model_id = model_id
        self._loaded_at = datetime.now(timezone.utc)
        self._ready_at = None
        self._last_used_at = None
        self._stopped_at = None
        self._failure_reason = None
        self._operation_count = 0
        self._error_count = 0

    def mark_ready(self) -> None:
        """Mark model as ready for operations. Transitions LOADING -> READY."""
        self._transition(ModelLifecycleState.READY, expected_states=[ModelLifecycleState.LOADING])
        self._ready_at = datetime.now(timezone.utc)

    def begin_use(self) -> None:
        """Mark model as busy performing an operation. Transitions READY -> BUSY."""
        self._transition(ModelLifecycleState.BUSY, expected_states=[ModelLifecycleState.READY])
        self._last_used_at = datetime.now(timezone.utc)
        self._operation_count += 1

    def end_use(self) -> None:
        """Mark model as completed operation and returned to READY. Transitions BUSY -> READY."""
        self._transition(ModelLifecycleState.READY, expected_states=[ModelLifecycleState.BUSY])

    def begin_stopping(self) -> None:
        """Begin stopping/unloading the model. Transitions READY/BUSY/FAILED -> STOPPING."""
        self._transition(
            ModelLifecycleState.STOPPING,
            expected_states=[
                ModelLifecycleState.READY,
                ModelLifecycleState.BUSY,
                ModelLifecycleState.FAILED,
            ],
        )

    def mark_unloaded(self) -> None:
        """Mark model as fully unloaded. Transitions STOPPING/FAILED -> UNLOADED."""
        if self._state == ModelLifecycleState.UNLOADED:
            return

        self._transition(
            ModelLifecycleState.UNLOADED,
            expected_states=[ModelLifecycleState.STOPPING, ModelLifecycleState.FAILED],
        )
        self._stopped_at = datetime.now(timezone.utc)
        self._model_id = None

    def mark_failed(self, reason: str) -> None:
        """Record model failure. Transitions LOADING/READY/BUSY/STOPPING -> FAILED."""
        if self._state == ModelLifecycleState.FAILED:
            self._failure_reason = reason
            return

        self._transition(
            ModelLifecycleState.FAILED,
            expected_states=[
                ModelLifecycleState.LOADING,
                ModelLifecycleState.READY,
                ModelLifecycleState.BUSY,
                ModelLifecycleState.STOPPING,
            ],
        )
        self._failure_reason = reason
        self._error_count += 1
        logger.warning("ModelLifecycle [%s] marked FAILED: %s", self._model_id or "none", reason)

    def sync_with_runtime(self, runtime: RuntimeAdapter) -> None:
        """Synchronize model lifecycle state with the process-level RuntimeAdapter.

        Detects unexpected crashes or failures reported by RuntimeAdapter.
        """
        runtime_is_alive = runtime.is_running()
        runtime_state = runtime.state

        # Case 1: Runtime explicitly failed
        if runtime_state == RuntimeState.FAILED:
            if self._state != ModelLifecycleState.FAILED and self._state != ModelLifecycleState.UNLOADED:
                err_msg = getattr(runtime, "last_stderr", None) or "Runtime reported FAILED state"
                self.mark_failed(err_msg)
            return

        # Case 2: Process unexpectedly terminated while lifecycle expects it active
        if not runtime_is_alive:
            if self._state in {ModelLifecycleState.READY, ModelLifecycleState.BUSY}:
                self.mark_failed("Runtime process terminated unexpectedly")
            elif self._state == ModelLifecycleState.STOPPING:
                self.mark_unloaded()

    def get_status(self, runtime: Optional[RuntimeAdapter] = None) -> ModelRuntimeStatus:
        """Return structured ModelRuntimeStatus snapshot."""
        if runtime is not None:
            self.sync_with_runtime(runtime)
            runtime_state = runtime.state
        else:
            runtime_state = RuntimeState.STOPPED

        return ModelRuntimeStatus(
            model_id=self._model_id,
            lifecycle_state=self._state,
            runtime_state=runtime_state,
            loaded_at=self._loaded_at,
            ready_at=self._ready_at,
            last_used_at=self._last_used_at,
            stopped_at=self._stopped_at,
            failure_reason=self._failure_reason,
            operation_count=self._operation_count,
            error_count=self._error_count,
        )
