"""Memory management and retrieval subsystem for WHIS."""

from app.memory.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL_ID,
    EmbeddingProvider,
    LlamaCppEmbeddingEngine,
    MockEmbeddingProvider,
)
from app.memory.exceptions import (
    DuplicateMemoryError,
    EmbeddingError,
    InvalidCategoryError,
    MemoryError,
    MemoryNotFoundError,
    StorageError,
    VectorStoreError,
)
from app.memory.manager import MemoryManager
from app.memory.models import MemoryCategory, MemoryRecord, MemorySearchResult
from app.memory.retrieval import MemoryRetriever
from app.memory.storage import MemoryStorage
from app.memory.vector_store import VectorStore, cosine_similarity

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DuplicateMemoryError",
    "EmbeddingError",
    "EmbeddingProvider",
    "InvalidCategoryError",
    "LlamaCppEmbeddingEngine",
    "MemoryCategory",
    "MemoryError",
    "MemoryManager",
    "MemoryNotFoundError",
    "MemoryRecord",
    "MemoryRetriever",
    "MemorySearchResult",
    "MemoryStorage",
    "MockEmbeddingProvider",
    "StorageError",
    "VectorStore",
    "VectorStoreError",
    "cosine_similarity",
]
