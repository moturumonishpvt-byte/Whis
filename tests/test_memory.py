"""Unit tests for WHIS Stage 5: Memory and RAG subsystem."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app.ai.inference import InferenceEngine
from app.ai.runtime import GenerationConfig, InferenceResult, RuntimeAdapter, RuntimeState
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
from app.memory.manager import MemoryManager
from app.memory.models import MemoryCategory, MemoryRecord, MemorySearchResult
from app.memory.retrieval import MemoryRetriever
from app.memory.storage import MemoryStorage
from app.memory.vector_store import VectorStore, cosine_similarity


class TestMemorySubsystem(unittest.TestCase):
    """Comprehensive test suite for Memory, VectorStore, Storage, and RAG."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "test_memory.db")
        self.embedding_provider = MockEmbeddingProvider(dimension=32)
        self.storage = MemoryStorage(db_path=self.db_path)
        self.vector_store = VectorStore(dimension=32)
        self.manager = MemoryManager(
            storage=self.storage,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )

    def tearDown(self) -> None:
        self.storage.close()
        self.tmp_dir.cleanup()

    def test_01_memory_creation(self) -> None:
        """1. remember() stores record with id, timestamps, vector, and category."""
        rec = self.manager.remember("WHIS is a local AI system.", category="PROJECT")

        self.assertIsNotNone(rec.id)
        self.assertEqual(rec.content, "WHIS is a local AI system.")
        self.assertEqual(rec.category, MemoryCategory.PROJECT)
        self.assertIsNotNone(rec.created_at)
        self.assertIsNotNone(rec.updated_at)
        self.assertIsNotNone(rec.embedding)
        self.assertEqual(len(rec.embedding), 32)
        self.assertEqual(self.vector_store.count(), 1)
        self.assertEqual(self.storage.count(), 1)

    def test_02_invalid_category_raises(self) -> None:
        """2. Specifying an invalid category raises InvalidCategoryError."""
        with self.assertRaises(InvalidCategoryError):
            self.manager.remember("Hello", category="NONEXISTENT_CATEGORY")

    def test_03_empty_content_raises(self) -> None:
        """3. Storing empty or whitespace-only content raises MemoryError."""
        with self.assertRaises(MemoryError):
            self.manager.remember("")
        with self.assertRaises(MemoryError):
            self.manager.remember("   \t\n  ")

    def test_04_get_existing_memory(self) -> None:
        """4. get() retrieves existing memory record by ID."""
        saved = self.manager.remember("Persistent facts", category="KNOWLEDGE")
        fetched = self.manager.get(saved.id)

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, saved.id)
        self.assertEqual(fetched.content, "Persistent facts")
        self.assertEqual(fetched.category, MemoryCategory.KNOWLEDGE)

    def test_05_get_nonexistent_memory(self) -> None:
        """5. get() returns None for unknown memory ID."""
        self.assertIsNone(self.manager.get("unknown-id-123"))

    def test_06_update_memory_content_reindexes(self) -> None:
        """6. Updating memory content regenerates embedding and updates vector store."""
        rec = self.manager.remember("Initial note", category="WORKING")
        old_emb = list(rec.embedding)

        updated = self.manager.update(rec.id, content="Updated note content")
        self.assertEqual(updated.content, "Updated note content")
        self.assertNotEqual(updated.embedding, old_emb)

        # Verify vector store has updated vector
        vec_in_store = self.vector_store.get(rec.id)
        self.assertEqual(vec_in_store, updated.embedding)

    def test_07_update_memory_metadata_and_category(self) -> None:
        """7. Updating metadata or category updates fields without re-embedding."""
        rec = self.manager.remember("User preference", category="PERSONAL", metadata={"author": "Alice"})
        updated = self.manager.update(rec.id, category="PROJECT", metadata={"priority": "high"})

        self.assertEqual(updated.category, MemoryCategory.PROJECT)
        self.assertEqual(updated.metadata.get("author"), "Alice")
        self.assertEqual(updated.metadata.get("priority"), "high")

    def test_08_update_missing_memory_raises(self) -> None:
        """8. Updating nonexistent memory raises MemoryNotFoundError."""
        with self.assertRaises(MemoryNotFoundError):
            self.manager.update("fake-id", content="New text")

    def test_09_delete_memory(self) -> None:
        """9. delete() removes memory from storage and vector index."""
        rec = self.manager.remember("To be deleted", category="WORKING")
        self.assertEqual(self.storage.count(), 1)
        self.assertEqual(self.vector_store.count(), 1)

        deleted = self.manager.delete(rec.id)
        self.assertTrue(deleted)
        self.assertEqual(self.storage.count(), 0)
        self.assertEqual(self.vector_store.count(), 0)
        self.assertIsNone(self.manager.get(rec.id))

    def test_10_delete_nonexistent_memory(self) -> None:
        """10. delete() returns False for nonexistent memory."""
        self.assertFalse(self.manager.delete("does-not-exist"))

    def test_11_persistence_hydration(self) -> None:
        """11. New MemoryManager hydrates vector store from existing SQLite database."""
        self.manager.remember("Architecture doc", category="PROJECT")
        self.manager.remember("User fact", category="PERSONAL")
        self.storage.close()

        # Create new manager pointing to same SQLite database file
        new_manager = MemoryManager(
            db_path=self.db_path,
            embedding_provider=self.embedding_provider,
        )
        self.assertEqual(new_manager.vector_store.count(), 2)
        results = new_manager.recall("Architecture", top_k=2)
        self.assertGreater(len(results), 0)

    def test_12_cosine_similarity_calculation(self) -> None:
        """12. cosine_similarity accurately computes scores for orthogonal and identical vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_13_cosine_similarity_dimension_mismatch(self) -> None:
        """13. Dimension mismatch raises VectorStoreError."""
        with self.assertRaises(VectorStoreError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_14_vector_store_add_and_search(self) -> None:
        """14. VectorStore correctly ranks top_k matches."""
        store = VectorStore(dimension=3)
        store.add("id1", [1.0, 0.0, 0.0])
        store.add("id2", [0.8, 0.2, 0.0])
        store.add("id3", [0.0, 1.0, 0.0])

        ranked = store.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0], "id1")
        self.assertEqual(ranked[1][0], "id2")

    def test_15_recall_semantic_ranking(self) -> None:
        """15. recall() returns sorted results with cosine similarity scores."""
        self.manager.remember("Python is a programming language", category="KNOWLEDGE")
        self.manager.remember("Apples are edible fruits", category="KNOWLEDGE")

        results = self.manager.recall("Python programming", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], MemorySearchResult)
        self.assertGreater(results[0].score, -1.0)
        self.assertLessEqual(results[0].score, 1.0)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_16_recall_category_filtering(self) -> None:
        """16. recall() restricts search candidates to the specified category."""
        self.manager.remember("Fix bug in backend", category="PROJECT")
        self.manager.remember("Today had a nice lunch", category="EPISODIC")

        results = self.manager.recall("backend", category="PROJECT")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.category, MemoryCategory.PROJECT)

    def test_17_duplicate_memory_deduplication(self) -> None:
        """17. allow_duplicates=False updates existing memory instead of inserting duplicate."""
        rec1 = self.manager.remember("Unique content", category="WORKING", allow_duplicates=False)
        rec2 = self.manager.remember("Unique content", category="WORKING", allow_duplicates=False)

        self.assertEqual(rec1.id, rec2.id)
        self.assertEqual(self.storage.count(), 1)
        self.assertEqual(self.vector_store.count(), 1)

    def test_18_duplicate_memory_raise(self) -> None:
        """18. raise_on_duplicate=True raises DuplicateMemoryError."""
        self.manager.remember("Strictly unique", category="KNOWLEDGE")
        with self.assertRaises(DuplicateMemoryError):
            self.manager.remember("Strictly unique", category="KNOWLEDGE", raise_on_duplicate=True)

    def test_19_clear_all_memories(self) -> None:
        """19. clear() deletes all records and clears the vector index."""
        self.manager.remember("Mem 1", category="WORKING")
        self.manager.remember("Mem 2", category="PERSONAL")
        self.assertEqual(self.storage.count(), 2)

        cleared_count = self.manager.clear()
        self.assertEqual(cleared_count, 2)
        self.assertEqual(self.storage.count(), 0)
        self.assertEqual(self.vector_store.count(), 0)

    def test_20_clear_category_memories(self) -> None:
        """20. clear(category) clears only target category."""
        self.manager.remember("Project task", category="PROJECT")
        self.manager.remember("Knowledge item", category="KNOWLEDGE")

        cleared_count = self.manager.clear(category="PROJECT")
        self.assertEqual(cleared_count, 1)
        self.assertEqual(self.storage.count(), 1)
        self.assertEqual(self.vector_store.count(), 1)
        self.assertEqual(self.manager.list_memories()[0].category, MemoryCategory.KNOWLEDGE)

    def test_21_embedding_failure_handling(self) -> None:
        """21. Embedding failure raises EmbeddingError without persisting record."""
        failing_provider = MagicMock(spec=EmbeddingProvider)
        failing_provider.embed.side_effect = EmbeddingError("Model out of memory")

        broken_manager = MemoryManager(
            storage=self.storage,
            embedding_provider=failing_provider,
        )
        with self.assertRaises(EmbeddingError):
            broken_manager.remember("Test prompt")
        self.assertEqual(self.storage.count(), 0)

    def test_22_storage_failure_handling(self) -> None:
        """22. Storage failures raise StorageError."""
        mock_storage = MagicMock(spec=MemoryStorage)
        mock_storage.list_all.return_value = []
        mock_storage.save.side_effect = StorageError("Disk I/O error")

        err_manager = MemoryManager(
            storage=mock_storage,
            embedding_provider=self.embedding_provider,
        )
        with self.assertRaises(StorageError):
            err_manager.remember("Valid content")

    def test_23_build_memory_context_rag(self) -> None:
        """23. build_memory_context() formats memories into structured context."""
        self.manager.remember("The user is building WHIS on Windows.", category="PROJECT")
        context_str = self.manager.build_memory_context("What is WHIS?", top_k=1)

        self.assertIn("Relevant Memories:", context_str)
        self.assertIn("[PROJECT] The user is building WHIS on Windows.", context_str)

    def test_24_memory_retriever_class(self) -> None:
        """24. MemoryRetriever exposes retrieve() and build_context()."""
        retriever = MemoryRetriever(self.manager)
        self.manager.remember("Offline inference is supported.", category="KNOWLEDGE")

        results = retriever.retrieve("offline", top_k=1)
        self.assertEqual(len(results), 1)

        context = retriever.build_context("offline", top_k=1)
        self.assertIn("Offline inference is supported.", context)

    def test_25_rag_inference_engine_integration(self) -> None:
        """25. Memory context integrates smoothly with Stage 4 InferenceEngine."""
        self.manager.remember("WHIS stands for Windows Home Intelligence System.", category="PROJECT")
        rag_context = self.manager.build_memory_context("What is WHIS?", top_k=1)

        # Mock runtime for InferenceEngine
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.generate.return_value = InferenceResult(
            text="WHIS is the Windows Home Intelligence System.",
            model_id="qwen3.5-4b",
            success=True,
            elapsed_seconds=0.1,
        )

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = True
        mock_loader._loaded_model_id = "qwen3.5-4b"

        engine = InferenceEngine(
            runtime=mock_runtime,
            loader=mock_loader,
        )

        # Pass RAG context into system prompt
        response = engine.ask(
            question="What does WHIS stand for?",
            system_prompt=f"Use the following memory:\n{rag_context}",
        )
        self.assertTrue(response.success)
        self.assertIn("WHIS", response.text)


if __name__ == "__main__":
    unittest.main()
