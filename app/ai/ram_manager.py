"""RAM-aware resource measurement and safety management for WHIS.

Provides system physical memory observation (total, available, used), configurable
safety reserves, approximate model memory estimation (weights, projector, context,
KV cache, runtime overhead), and load safety evaluations (SAFE, WARNING, UNSAFE)
without loading model tensors or initiating llama.cpp subprocesses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, Union

import psutil

from app.ai.model_manager import ModelDefinition, ModelRegistry
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)

# Default safety reserve preserved for Windows OS and applications: 2.0 GB
DEFAULT_SAFETY_RESERVE_BYTES = 2 * 1024 * 1024 * 1024


class MemoryDecision(str, Enum):
    """Evaluation outcome for whether a model can be safely loaded into RAM."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    UNSAFE = "UNSAFE"


class RAMManagerError(WHISError):
    """Base exception for RAM Manager failures."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id


class InsufficientMemoryError(RAMManagerError):
    """Raised when a model load is rejected due to unsafe memory requirements."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        evaluation: Optional[SafetyEvaluation] = None,
    ) -> None:
        super().__init__(message, model_id=model_id)
        self.evaluation = evaluation


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable snapshot of physical system memory at a specific instant."""

    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float
    timestamp: datetime

    @property
    def total_gb(self) -> float:
        """Total physical memory in gigabytes."""
        return round(self.total_bytes / (1024**3), 2)

    @property
    def available_gb(self) -> float:
        """Available memory in gigabytes that can be given immediately to processes."""
        return round(self.available_bytes / (1024**3), 2)

    @property
    def used_gb(self) -> float:
        """Used memory in gigabytes (total - available)."""
        return round(self.used_bytes / (1024**3), 2)


@dataclass(frozen=True)
class ModelMemoryEstimate:
    """Structured approximation of runtime memory required by a model."""

    model_id: str
    weight_bytes: int
    projector_bytes: int
    context_tokens: int
    estimated_kv_cache_bytes: int
    estimated_runtime_overhead_bytes: int
    estimated_total_bytes: int
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    notes: str

    @property
    def weight_gb(self) -> float:
        """Model weight size in gigabytes."""
        return round(self.weight_bytes / (1024**3), 2)

    @property
    def estimated_total_gb(self) -> float:
        """Total estimated runtime requirement in gigabytes."""
        return round(self.estimated_total_bytes / (1024**3), 2)


@dataclass(frozen=True)
class SafetyEvaluation:
    """Detailed safety evaluation regarding whether a model load fits within safe RAM."""

    decision: MemoryDecision
    model_id: str
    available_bytes: int
    safety_reserve_bytes: int
    safe_available_bytes: int
    estimated_model_bytes: int
    remaining_after_load_bytes: int
    reason: str

    @property
    def is_safe(self) -> bool:
        """Return True if decision is SAFE."""
        return self.decision == MemoryDecision.SAFE

    @property
    def is_executable(self) -> bool:
        """Return True if decision is SAFE or WARNING (executable with caution)."""
        return self.decision in {MemoryDecision.SAFE, MemoryDecision.WARNING}


class MemoryProvider(ABC):
    """Abstract provider for observing system physical memory."""

    @abstractmethod
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Capture and return current memory snapshot."""

    @abstractmethod
    def get_process_memory(self, pid: int) -> Optional[int]:
        """Return Resident Set Size (RSS) memory in bytes for process, if available."""


class SystemMemoryProvider(MemoryProvider):
    """System memory provider using psutil for reliable cross-platform observation."""

    def get_memory_snapshot(self) -> MemorySnapshot:
        vm = psutil.virtual_memory()
        total = vm.total
        available = vm.available
        used = total - available
        percent = (used / total) * 100.0 if total > 0 else 0.0

        return MemorySnapshot(
            total_bytes=total,
            available_bytes=available,
            used_bytes=used,
            percent_used=round(percent, 2),
            timestamp=datetime.now(timezone.utc),
        )

    def get_process_memory(self, pid: int) -> Optional[int]:
        try:
            proc = psutil.Process(pid)
            return proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class RAMManager:
    """RAM-aware resource manager for WHIS.

    Measures system physical RAM, estimates model requirements without executing models,
    and evaluates whether loading a model is safe against a configurable safety reserve.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        memory_provider: Optional[MemoryProvider] = None,
        safety_reserve_bytes: int = DEFAULT_SAFETY_RESERVE_BYTES,
    ) -> None:
        self.registry: ModelRegistry = registry if registry is not None else ModelRegistry()
        self.memory_provider: MemoryProvider = (
            memory_provider if memory_provider is not None else SystemMemoryProvider()
        )
        self.safety_reserve_bytes: int = max(0, safety_reserve_bytes)

    def get_system_memory(self) -> MemorySnapshot:
        """Return the current observed system memory snapshot."""
        return self.memory_provider.get_memory_snapshot()

    def get_process_memory(self, pid: int) -> Optional[int]:
        """Return the observed memory consumption of a process in bytes."""
        return self.memory_provider.get_process_memory(pid)

    def estimate_model_memory(
        self,
        model_id: str,
        context_tokens: Optional[int] = None,
    ) -> ModelMemoryEstimate:
        """Compute an approximate runtime memory estimate for a registered model.

        Calculates:
        1. Weight bytes: Inspected from GGUF file on disk (or registry size_gb fallback).
        2. Projector bytes: Inspected from mmproj file on disk if multimodal.
        3. Context KV cache: Conservative approximation based on context tokens and model scale.
        4. Runtime overhead: Scratch allocations, compute graph buffers, and runtime baseline.

        Does NOT load weights into RAM or invoke llama.cpp.

        Args:
            model_id: Registered model identifier.
            context_tokens: Context size in tokens (defaults to model.default_context).

        Returns:
            ModelMemoryEstimate detailing the calculated components and confidence.
        """
        model = self.registry.get(model_id)
        effective_context = (
            context_tokens if context_tokens is not None and context_tokens > 0 else model.default_context
        )

        # 1. Weight file size
        weight_bytes = 0
        try:
            model_path = self.registry.resolve_model_path(model_id)
            if model_path.is_file():
                weight_bytes = model_path.stat().st_size
        except Exception:
            pass

        if weight_bytes == 0 and model.size_gb is not None:
            weight_bytes = int(model.size_gb * (1024**3))

        # 2. Projector file size
        projector_bytes = 0
        if model.has_projector and model.mmproj_path:
            try:
                mmproj_path = self.registry.resolve_mmproj_path(model_id)
                if mmproj_path and mmproj_path.is_file():
                    projector_bytes = mmproj_path.stat().st_size
            except Exception:
                pass

        # 3. KV Cache approximation
        # Conservative approximation:
        # Scale bytes-per-token by model parameter scale (inferred from weight size):
        # Base scale: ~64 KB per token for small models, up to ~256 KB per token for 30B+
        if weight_bytes > 10 * 1024 * 1024 * 1024:  # > 10 GB (e.g. 30B models)
            bytes_per_token = 256 * 1024
            confidence = "MEDIUM"
        elif weight_bytes > 2 * 1024 * 1024 * 1024:  # 2 GB - 10 GB (e.g. 4B - 7B models)
            bytes_per_token = 128 * 1024
            confidence = "MEDIUM"
        else:  # Small models (< 2 GB, e.g. 1.2B)
            bytes_per_token = 64 * 1024
            confidence = "MEDIUM"

        estimated_kv_cache_bytes = effective_context * bytes_per_token

        # 4. Runtime overhead (scratch buffers, compute graphs, llama.cpp baseline: 300 MB + 5% weights)
        estimated_runtime_overhead_bytes = 300 * 1024 * 1024 + int(weight_bytes * 0.05)

        # 5. Total
        estimated_total_bytes = (
            weight_bytes
            + projector_bytes
            + estimated_kv_cache_bytes
            + estimated_runtime_overhead_bytes
        )

        notes = (
            f"Approximation based on {round(weight_bytes / (1024**3), 2)} GB weights, "
            f"{round(projector_bytes / (1024**2), 1)} MB projector, {effective_context} tokens context "
            f"(~{round(estimated_kv_cache_bytes / (1024**2), 1)} MB KV cache), and "
            f"~{round(estimated_runtime_overhead_bytes / (1024**2), 1)} MB runtime overhead."
        )

        return ModelMemoryEstimate(
            model_id=model_id,
            weight_bytes=weight_bytes,
            projector_bytes=projector_bytes,
            context_tokens=effective_context,
            estimated_kv_cache_bytes=estimated_kv_cache_bytes,
            estimated_runtime_overhead_bytes=estimated_runtime_overhead_bytes,
            estimated_total_bytes=estimated_total_bytes,
            confidence=confidence,
            notes=notes,
        )

    def evaluate_safety(
        self,
        model_id: str,
        context_tokens: Optional[int] = None,
    ) -> SafetyEvaluation:
        """Evaluate whether a model can be safely loaded given current RAM and safety reserve.

        Decisions:
        - SAFE: Estimated model RAM requirement <= (available RAM - safety reserve).
        - WARNING: Estimated model RAM requirement fits in available RAM, but encroaches on safety reserve.
        - UNSAFE: Estimated model RAM requirement exceeds total available RAM.

        Args:
            model_id: Registered model identifier.
            context_tokens: Optional custom context size in tokens.

        Returns:
            SafetyEvaluation with decision, breakdown, and descriptive explanation.
        """
        snapshot = self.get_system_memory()
        estimate = self.estimate_model_memory(model_id, context_tokens=context_tokens)

        available_bytes = snapshot.available_bytes
        reserve_bytes = self.safety_reserve_bytes
        safe_available_bytes = max(0, available_bytes - reserve_bytes)
        estimated_model_bytes = estimate.estimated_total_bytes

        remaining_after_load_bytes = safe_available_bytes - estimated_model_bytes

        safe_avail_gb = round(safe_available_bytes / (1024**3), 2)
        avail_gb = round(available_bytes / (1024**3), 2)
        reserve_gb = round(reserve_bytes / (1024**3), 2)
        est_gb = round(estimated_model_bytes / (1024**3), 2)

        if estimated_model_bytes <= safe_available_bytes:
            decision = MemoryDecision.SAFE
            reason = (
                f"Estimated requirement ({est_gb} GB) fits comfortably within safe available RAM "
                f"({safe_avail_gb} GB remaining after preserving {reserve_gb} GB safety reserve)."
            )
        elif estimated_model_bytes <= available_bytes:
            decision = MemoryDecision.WARNING
            encroachment = round((estimated_model_bytes - safe_available_bytes) / (1024**3), 2)
            reason = (
                f"Estimated requirement ({est_gb} GB) fits within available RAM ({avail_gb} GB) "
                f"but encroaches on the {reserve_gb} GB safety reserve by {encroachment} GB."
            )
        else:
            decision = MemoryDecision.UNSAFE
            deficit = round((estimated_model_bytes - available_bytes) / (1024**3), 2)
            reason = (
                f"Estimated requirement ({est_gb} GB) exceeds total available RAM ({avail_gb} GB) "
                f"by {deficit} GB. Attempting to load would likely cause memory exhaustion or paging."
            )

        logger.info("RAMManager evaluation for '%s': %s (%s)", model_id, decision.value, reason)

        return SafetyEvaluation(
            decision=decision,
            model_id=model_id,
            available_bytes=available_bytes,
            safety_reserve_bytes=reserve_bytes,
            safe_available_bytes=safe_available_bytes,
            estimated_model_bytes=estimated_model_bytes,
            remaining_after_load_bytes=remaining_after_load_bytes,
            reason=reason,
        )

    def can_load(
        self,
        model_id: str,
        context_tokens: Optional[int] = None,
        allow_warning: bool = True,
    ) -> bool:
        """Determine whether a model can be safely loaded.

        Args:
            model_id: Registered model identifier.
            context_tokens: Optional custom context size in tokens.
            allow_warning: If True, WARNING decisions are permitted; if False, only SAFE is allowed.

        Returns:
            True if memory conditions permit loading, False otherwise.
        """
        evaluation = self.evaluate_safety(model_id, context_tokens=context_tokens)
        if evaluation.decision == MemoryDecision.SAFE:
            return True
        if evaluation.decision == MemoryDecision.WARNING:
            return allow_warning
        return False
