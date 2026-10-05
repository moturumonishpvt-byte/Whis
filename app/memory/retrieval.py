"""Retrieval abstraction for semantic context construction (RAG)."""

from __future__ import annotations

from typing import List, Optional, Union

from app.memory.manager import MemoryManager
from app.memory.models import MemoryCategory, MemorySearchResult


class MemoryRetriever:
    """Retrieves relevant memories and constructs formatted context for Stage 4 inference."""

    def __init__(self, manager: MemoryManager) -> None:
        self.manager: MemoryManager = manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> List[MemorySearchResult]:
        """Perform semantic retrieval using the underlying MemoryManager."""
        return self.manager.recall(query=query, top_k=top_k, category=category)

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[Union[str, MemoryCategory]] = None,
        header: str = "Relevant Memories:",
    ) -> str:
        """Build structured memory context block suitable for prompt injection."""
        return self.manager.build_memory_context(
            query=query,
            top_k=top_k,
            category=category,
            header=header,
        )
