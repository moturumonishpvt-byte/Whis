"""Embedding generation providers for WHIS memory subsystem."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import logging
import math
from typing import List, Optional

from app.ai.loader import ModelLoader
from app.ai.model_manager import ModelRegistry
from app.ai.runtime import LlamaCppRuntime, RuntimeAdapter, RuntimeAdapterError
from app.memory.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID: str = "qwen3-embedding-4b"
DEFAULT_EMBEDDING_DIM: int = 2560


class EmbeddingProvider(ABC):
    """Abstract base class for memory embedding generation."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate a dense embedding vector for a given text."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return [self.embed(t) for t in texts]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, fast mock embedding provider for tests and offline usage.

    Produces normalized, reproducible float vectors based on SHA-256 hash
    without requiring any model execution or network calls.
    """

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")
        self.dimension: int = dimension

    def embed(self, text: str) -> List[float]:
        """Return a deterministic normalized vector derived from text hash."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text.")

        # Seed with SHA-256 hash of text
        h = hashlib.sha256(text.strip().encode("utf-8")).digest()
        raw_vals: List[float] = []

        for i in range(self.dimension):
            # Mix hash bytes deterministically
            byte_val = h[i % len(h)]
            float_val = float((byte_val ^ (i & 0xFF)) - 128) / 128.0
            raw_vals.append(float_val)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in raw_vals))
        if norm <= 0.0:
            return [1.0 / math.sqrt(self.dimension)] * self.dimension
        return [x / norm for x in raw_vals]


class LlamaCppEmbeddingEngine(EmbeddingProvider):
    """Integrates Qwen3-Embedding-4B through WHIS ModelLoader & RuntimeAdapter.

    Obeys the single-active-model policy and Stage 3 lifecycle state machine.
    """

    def __init__(
        self,
        loader: Optional[ModelLoader] = None,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        auto_unload: bool = False,
    ) -> None:
        self.model_id: str = model_id
        self.auto_unload: bool = auto_unload

        if loader is not None:
            self.loader: ModelLoader = loader
        else:
            registry = ModelRegistry()
            runtime = LlamaCppRuntime()
            self.loader = ModelLoader(registry=registry, runtime=runtime)

    def _ensure_loaded(self) -> None:
        """Ensure the embedding model is loaded into the runtime."""
        if self.loader.is_loaded(self.model_id):
            return

        # If another model is active, ModelLoader.load() will require unloading it first
        if self.loader.is_loaded():
            logger.info("Unloading currently active model before loading embedding model '%s'.", self.model_id)
            self.loader.unload()

        logger.info("Loading embedding model '%s'.", self.model_id)
        try:
            self.loader.load(self.model_id)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_id}': {exc}"
            ) from exc

    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector using Qwen3-Embedding-4B.

        Raises:
            EmbeddingError: If text is empty or runtime embedding fails.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text.")

        self._ensure_loaded()

        try:
            model_def = self.loader.loaded_model()
            if model_def is None:
                raise EmbeddingError(f"Embedding model '{self.model_id}' is not loaded.")

            vec = self.loader.runtime.embed(model_def, text.strip())
            if not vec:
                raise EmbeddingError(f"Embedding model returned empty vector for text.")
            return vec
        except RuntimeAdapterError as exc:
            raise EmbeddingError(
                f"Runtime embedding generation failed for model '{self.model_id}': {exc}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Unexpected embedding error for model '{self.model_id}': {exc}"
            ) from exc
        finally:
            if self.auto_unload:
                self.unload()

    def unload(self) -> None:
        """Unload the embedding model from runtime if loaded."""
        if self.loader.is_loaded(self.model_id):
            logger.info("Unloading embedding model '%s'.", self.model_id)
            self.loader.unload()
