"""Central MemoryManager coordinating storage, embeddings, indexing, and retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from app.memory.embeddings import EmbeddingProvider, MockEmbeddingProvider
from app.memory.exceptions import (
    DuplicateMemoryError,
    EmbeddingError,
    InvalidCategoryError,
    MemoryError,
    MemoryNotFoundError,
    StorageError,
    VectorStoreError,
)
from app.memory.models import MemoryCategory, MemoryRecord, MemorySearchResult
from app.memory.storage import MemoryStorage
from app.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates persistent memory storage, vector indexing, and semantic retrieval."""

    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.storage: MemoryStorage = (
            storage if storage is not None else MemoryStorage(db_path=db_path)
        )
        self.vector_store: VectorStore = (
            vector_store if vector_store is not None else VectorStore()
        )
        self.embedding_provider: EmbeddingProvider = (
            embedding_provider if embedding_provider is not None else MockEmbeddingProvider()
        )

        # Hydrate vector store with existing records from persistent storage
        self._hydrate_vectors()

    def _hydrate_vectors(self) -> None:
        """Load stored embeddings from database into in-memory vector index."""
        try:
            records = self.storage.list_all()
            for rec in records:
                if rec.embedding:
                    self.vector_store.add(rec.id, rec.embedding)
        except Exception as exc:
            logger.warning("Could not hydrate vector store on startup: %s", exc)

    def remember(
        self,
        content: str,
        category: Union[str, MemoryCategory] = MemoryCategory.WORKING,
        metadata: Optional[Dict[str, Any]] = None,
        allow_duplicates: bool = True,
        raise_on_duplicate: bool = False,
    ) -> MemoryRecord:
        """Store a new memory record, generate its embedding, and index its vector.

        Args:
            content: Non-empty textual content to remember.
            category: One of the supported memory categories.
            metadata: Optional dictionary of extra attributes.
            allow_duplicates: If False, deduplicates identical content in the same category.
            raise_on_duplicate: If True and duplicate detected, raises DuplicateMemoryError.

        Returns:
            The created (or existing deduplicated) MemoryRecord.

        Raises:
            MemoryError: If content is empty or invalid.
            InvalidCategoryError: If category is unrecognized.
            DuplicateMemoryError: If raise_on_duplicate is True and content exists.
            EmbeddingError: If embedding generation fails.
            StorageError: If database persistence fails.
        """
        if not content or not content.strip():
            raise MemoryError("Memory content cannot be empty.")

        cat_enum = MemoryCategory.from_str(category)
        clean_content = content.strip()

        # Check duplicate
        if not allow_duplicates or raise_on_duplicate:
            existing = self.storage.find_by_content(clean_content, category=cat_enum)
            if existing is not None:
                if raise_on_duplicate:
                    raise DuplicateMemoryError(
                        f"Duplicate memory already exists in category '{cat_enum.value}'.",
                        memory_id=existing.id,
                    )
                # Deduplicate: update metadata and timestamp
                if metadata:
                    existing.metadata.update(metadata)
                existing.touch()
                self.storage.update(existing)
                return existing

        # Generate embedding vector
        try:
            embedding = self.embedding_provider.embed(clean_content)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        memory_id = str(uuid.uuid4())
        record = MemoryRecord.create(
            memory_id=memory_id,
            content=clean_content,
            category=cat_enum,
            embedding=embedding,
            metadata=metadata or {},
        )

        # Persist to database
        self.storage.save(record)

        # Index vector
        try:
            if embedding:
                self.vector_store.add(record.id, embedding)
        except Exception as exc:
            # Rollback storage on vector failure
            self.storage.delete(record.id)
            raise VectorStoreError(f"Failed to index memory vector: {exc}", memory_id=record.id) from exc

        logger.info("Remembered memory '%s' in category '%s'.", record.id, cat_enum.value)
        return record

    def recall(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> List[MemorySearchResult]:
        """Semantically search memories relevant to the query.

        Args:
            query: Non-empty search question or phrase.
            top_k: Max number of results to return.
            category: Optional category filter.

        Returns:
            List of MemorySearchResult ordered by relevance score descending.

        Raises:
            MemoryError: If query is empty.
            InvalidCategoryError: If category is unrecognized.
            EmbeddingError: If query embedding generation fails.
        """
        if not query or not query.strip():
            raise MemoryError("Search query cannot be empty.")
        if top_k <= 0:
            return []

        # Validate category if provided
        candidate_ids: Optional[Set[str]] = None
        if category is not None:
            cat_enum = MemoryCategory.from_str(category)
            category_records = self.storage.list_all(category=cat_enum)
            candidate_ids = {r.id for r in category_records}
            if not candidate_ids:
                return []

        # Embed query
        try:
            query_vector = self.embedding_provider.embed(query.strip())
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Query embedding generation failed: {exc}") from exc

        # Search vector store
        ranked = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            candidate_ids=candidate_ids,
        )

        results: List[MemorySearchResult] = []
        for mem_id, score in ranked:
            rec = self.storage.get(mem_id)
            if rec is not None:
                results.append(MemorySearchResult(record=rec, score=score))

        return results

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch a single memory by ID."""
        if not memory_id:
            return None
        return self.storage.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        category: Optional[Union[str, MemoryCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Update an existing memory record.

        If content changes, the embedding is re-generated and re-indexed.

        Raises:
            MemoryNotFoundError: If memory_id does not exist.
            MemoryError: If content is updated to empty string.
            InvalidCategoryError: If category is invalid.
        """
        record = self.storage.get(memory_id)
        if record is None:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found.", memory_id=memory_id)

        content_changed = False
        if content is not None:
            clean = content.strip()
            if not clean:
                raise MemoryError("Updated memory content cannot be empty.")
            if clean != record.content:
                record.content = clean
                content_changed = True

        if category is not None:
            record.category = MemoryCategory.from_str(category)

        if metadata is not None:
            record.metadata.update(metadata)

        if content_changed:
            new_embedding = self.embedding_provider.embed(record.content)
            record.embedding = new_embedding
            self.vector_store.update(record.id, new_embedding)

        self.storage.update(record)
        return record

    def delete(self, memory_id: str) -> bool:
        """Delete memory from persistent storage and vector store."""
        if not memory_id:
            return False
        self.vector_store.remove(memory_id)
        return self.storage.delete(memory_id)

    def list_memories(
        self,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> List[MemoryRecord]:
        """List all memories, optionally filtered by category."""
        return self.storage.list_all(category=category)

    def clear(
        self,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> int:
        """Clear memories from storage and vector store. Returns count cleared."""
        if category is not None:
            cat_enum = MemoryCategory.from_str(category)
            records = self.storage.list_all(category=cat_enum)
            ids = {r.id for r in records}
            self.vector_store.clear(ids=ids)
            return self.storage.clear(category=cat_enum)
        else:
            self.vector_store.clear()
            return self.storage.clear()

    def build_memory_context(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[Union[str, MemoryCategory]] = None,
        header: str = "Relevant Memories:",
    ) -> str:
        """Format top-k retrieved memories into context text for Stage 4 inference."""
        results = self.recall(query=query, top_k=top_k, category=category)
        if not results:
            return ""

        lines = [header]
        for res in results:
            lines.append(f"- [{res.record.category.value}] {res.record.content}")
        return "\n".join(lines)
