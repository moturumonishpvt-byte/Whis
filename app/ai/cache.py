"""Model cache abstraction for WHIS.

Maintains in-memory metadata regarding recently used models, usage counts,
last-used timestamps (LRU), and estimated memory footprint without keeping
large GGUF model tensors resident in RAM or managing OS subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from threading import RLock
from typing import Any, Dict, List, Optional

from app.ai.lifecycle import ModelLifecycleState
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)

DEFAULT_MAX_CACHE_ENTRIES = 5


class CacheState(str, Enum):
    """Status of a model entry in the cache."""

    NOT_CACHED = "NOT_CACHED"
    CACHED_METADATA = "CACHED_METADATA"
    ACTIVE = "ACTIVE"


class ModelCacheError(WHISError):
    """Base exception for model cache operations."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.model_id = model_id


@dataclass
class ModelCacheEntry:
    """Metadata representing a cached model."""

    model_id: str
    last_used_at: datetime
    last_loaded_at: Optional[datetime] = None
    usage_count: int = 1
    estimated_memory_bytes: int = 0
    lifecycle_state: Optional[ModelLifecycleState] = None
    cache_state: CacheState = CacheState.CACHED_METADATA
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Return True if model is currently marked as the active runtime model."""
        return self.cache_state == CacheState.ACTIVE

    @property
    def estimated_memory_gb(self) -> float:
        """Estimated memory requirement in gigabytes."""
        return round(self.estimated_memory_bytes / (1024**3), 2)


class ModelCache:
    """Thread-safe in-memory model metadata cache with LRU tracking.

    Enables WHIS to track model reuse history, execution frequency, and
    memory requirements across model lifecycles without loading weights into RAM.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_CACHE_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries: int = max_entries
        self._entries: Dict[str, ModelCacheEntry] = {}
        self._lock: RLock = RLock()

    def size(self) -> int:
        """Return the number of entries currently stored in cache."""
        with self._lock:
            return len(self._entries)

    def __len__(self) -> int:
        return self.size()

    def contains(self, model_id: str) -> bool:
        """Check whether metadata for model_id exists in the cache."""
        with self._lock:
            return model_id in self._entries

    def __contains__(self, model_id: str) -> bool:
        return self.contains(model_id)

    def get(self, model_id: str) -> Optional[ModelCacheEntry]:
        """Retrieve the cached entry for model_id, or None if not found."""
        with self._lock:
            return self._entries.get(model_id)

    def put(self, entry: ModelCacheEntry) -> None:
        """Add or update an entry in the cache.

        If capacity is reached, evicts the least recently used entry that is NOT currently ACTIVE.
        """
        with self._lock:
            if entry.model_id in self._entries:
                self._entries[entry.model_id] = entry
                return

            if len(self._entries) >= self.max_entries:
                lru_entry = self._get_lru_evictable()
                if lru_entry is not None:
                    del self._entries[lru_entry.model_id]
                    logger.debug("ModelCache evicted LRU entry '%s'.", lru_entry.model_id)

            self._entries[entry.model_id] = entry

    def touch(self, model_id: str) -> Optional[ModelCacheEntry]:
        """Update last_used_at timestamp of a cached entry to current UTC time."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                entry.last_used_at = datetime.now(timezone.utc)
            return entry

    def increment_usage(self, model_id: str) -> Optional[ModelCacheEntry]:
        """Increment usage_count and refresh last_used_at timestamp."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                entry.usage_count += 1
                entry.last_used_at = datetime.now(timezone.utc)
            return entry

    def remove(self, model_id: str) -> Optional[ModelCacheEntry]:
        """Remove and return entry for model_id, or None if not present."""
        with self._lock:
            return self._entries.pop(model_id, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._entries.clear()

    def list_entries(self) -> List[ModelCacheEntry]:
        """Return list of all cached entries sorted by last_used_at descending."""
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda e: e.last_used_at,
                reverse=True,
            )

    def get_least_recently_used(self) -> Optional[ModelCacheEntry]:
        """Return the entry with the oldest last_used_at timestamp, or None if empty."""
        with self._lock:
            if not self._entries:
                return None
            return min(self._entries.values(), key=lambda e: e.last_used_at)

    def _get_lru_evictable(self) -> Optional[ModelCacheEntry]:
        """Identify LRU entry prioritizing non-active entries for eviction."""
        evictable = [e for e in self._entries.values() if not e.is_active]
        if not evictable:
            return min(self._entries.values(), key=lambda e: e.last_used_at)
        return min(evictable, key=lambda e: e.last_used_at)

    def total_estimated_cached_memory(self) -> int:
        """Return the sum of estimated memory across all cached entries in bytes."""
        with self._lock:
            return sum(e.estimated_memory_bytes for e in self._entries.values())
