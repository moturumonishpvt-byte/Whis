"""Exceptions for WHIS memory and retrieval subsystem."""

from __future__ import annotations

from typing import Optional

from app.core.exceptions import WHISError


class MemoryError(WHISError):
    """Base exception for all memory and retrieval errors."""

    def __init__(self, message: str, memory_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.memory_id = memory_id


class InvalidCategoryError(MemoryError):
    """Raised when an invalid or unsupported memory category is specified."""


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory record does not exist."""


class StorageError(MemoryError):
    """Raised when a persistent storage operation fails."""


class EmbeddingError(MemoryError):
    """Raised when embedding generation fails."""


class VectorStoreError(MemoryError):
    """Raised when a vector store indexing or retrieval operation fails."""


class DuplicateMemoryError(MemoryError):
    """Raised when attempting to store duplicate memory content when prohibited."""
