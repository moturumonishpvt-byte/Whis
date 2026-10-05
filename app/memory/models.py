"""Memory records and category data structures for WHIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from app.memory.exceptions import InvalidCategoryError


class MemoryCategory(str, Enum):
    """Supported memory categories in WHIS."""

    WORKING = "WORKING"
    EPISODIC = "EPISODIC"
    PERSONAL = "PERSONAL"
    PROJECT = "PROJECT"
    KNOWLEDGE = "KNOWLEDGE"

    @classmethod
    def from_str(cls, val: Union[str, MemoryCategory]) -> MemoryCategory:
        """Parse and validate a memory category."""
        if isinstance(val, MemoryCategory):
            return val
        if not isinstance(val, str):
            raise InvalidCategoryError(f"Category must be a string or MemoryCategory, got {type(val).__name__}")
        normalized = val.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = [c.value for c in cls]
            raise InvalidCategoryError(
                f"Invalid memory category '{val}'. Must be one of: {', '.join(valid)}"
            ) from exc


@dataclass
class MemoryRecord:
    """A single persistent memory record."""

    id: str
    content: str
    category: MemoryCategory
    created_at: str  # ISO 8601 UTC timestamp
    updated_at: str  # ISO 8601 UTC timestamp
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.category, MemoryCategory):
            self.category = MemoryCategory.from_str(self.category)

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        category: Union[str, MemoryCategory],
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Factory method with current timestamp."""
        now_iso = datetime.now(timezone.utc).isoformat()
        return cls(
            id=memory_id,
            content=content,
            category=MemoryCategory.from_str(category),
            created_at=now_iso,
            updated_at=now_iso,
            embedding=embedding,
            metadata=metadata or {},
        )

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemorySearchResult:
    """A scored memory item returned from semantic retrieval."""

    record: MemoryRecord
    score: float  # Cosine similarity score between -1.0 and 1.0
