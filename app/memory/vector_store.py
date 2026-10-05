"""In-process lightweight vector store for semantic memory search."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from app.memory.exceptions import VectorStoreError


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns a score in [-1.0, 1.0]. Returns 0.0 if either vector has zero magnitude.
    Raises VectorStoreError if lengths do not match.
    """
    if len(vec1) != len(vec2):
        raise VectorStoreError(
            f"Vector dimension mismatch: query vector has length {len(vec1)}, "
            f"stored vector has length {len(vec2)}"
        )

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class VectorStore:
    """In-process indexed vector store for embedding vectors."""

    def __init__(self, dimension: Optional[int] = None) -> None:
        self.dimension: Optional[int] = dimension
        self._vectors: Dict[str, List[float]] = {}

    def add(self, memory_id: str, vector: List[float]) -> None:
        """Add a vector for the specified memory_id.

        Raises:
            VectorStoreError: If vector is empty or dimension mismatches configured dimension.
        """
        if not vector:
            raise VectorStoreError(f"Cannot add empty vector for memory '{memory_id}'", memory_id=memory_id)

        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch for memory '{memory_id}': expected {self.dimension}, got {len(vector)}",
                memory_id=memory_id,
            )

        self._vectors[memory_id] = list(vector)

    def update(self, memory_id: str, vector: List[float]) -> None:
        """Update the vector for an existing memory ID."""
        self.add(memory_id, vector)

    def remove(self, memory_id: str) -> bool:
        """Remove a vector by memory ID. Returns True if removed, False if not present."""
        if memory_id in self._vectors:
            del self._vectors[memory_id]
            return True
        return False

    def get(self, memory_id: str) -> Optional[List[float]]:
        """Return the vector for the given memory ID, if present."""
        vec = self._vectors.get(memory_id)
        return list(vec) if vec is not None else None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform cosine similarity search against stored vectors.

        Args:
            query_vector: Target embedding vector.
            top_k: Maximum number of results to return.
            candidate_ids: Optional set of allowed IDs (e.g., filtered by category).

        Returns:
            List of (memory_id, similarity_score) tuples, sorted descending by score.
        """
        if not query_vector:
            raise VectorStoreError("Query vector cannot be empty.")
        if top_k <= 0:
            return []

        if self.dimension is not None and len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        scored: List[Tuple[str, float]] = []

        for mem_id, vec in self._vectors.items():
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue

            score = cosine_similarity(query_vector, vec)
            scored.append((mem_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return number of vectors in store."""
        return len(self._vectors)

    def clear(self, ids: Optional[Set[str]] = None) -> None:
        """Clear all vectors, or only those matching ids."""
        if ids is None:
            self._vectors.clear()
        else:
            for mem_id in ids:
                self._vectors.pop(mem_id, None)
