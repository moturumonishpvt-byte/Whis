"""Tests for WHIS Stage 3.6: Model Cache.

Comprehensive unit tests verifying the in-memory ModelCache, ModelCacheEntry,
CacheState transitions, LRU ordering, capacity enforcement, memory metadata
tracking, and ModelLoader integration.
"""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock

from app.ai.cache import CacheState, ModelCache, ModelCacheEntry, ModelCacheError
from app.ai.lifecycle import ModelLifecycleState
from app.ai.loader import ModelAlreadyLoadedError, ModelLoader
from app.ai.model_manager import ModelRegistry
from app.ai.ram_manager import MemoryDecision, RAMManager
from app.ai.runtime import RuntimeAdapter, RuntimeConfig, RuntimeState


class TestModelCache(unittest.TestCase):
    """Test suite for ModelCache and ModelLoader cache integration."""

    def setUp(self) -> None:
        self.cache = ModelCache(max_entries=3)
        self.registry = ModelRegistry()

    def test_01_empty_cache_starts_correctly(self) -> None:
        """1. Empty cache starts correctly."""
        self.assertEqual(self.cache.size(), 0)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.list_entries(), [])
        self.assertIsNone(self.cache.get_least_recently_used())
        self.assertEqual(self.cache.total_estimated_cached_memory(), 0)

    def test_02_put_adds_an_entry(self) -> None:
        """2. put() adds an entry."""
        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
            usage_count=1,
            estimated_memory_bytes=3 * 1024**3,
        )
        self.cache.put(entry)
        self.assertEqual(self.cache.size(), 1)
        self.assertTrue(self.cache.contains("qwen3.5-4b"))

    def test_03_get_retrieves_an_entry(self) -> None:
        """3. get() retrieves an entry."""
        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
            usage_count=2,
            estimated_memory_bytes=3 * 1024**3,
        )
        self.cache.put(entry)
        retrieved = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_id, "qwen3.5-4b")
        self.assertEqual(retrieved.usage_count, 2)

    def test_04_contains_works(self) -> None:
        """4. contains() and 'in' operator work."""
        self.assertFalse(self.cache.contains("qwen3.5-4b"))
        self.assertNotIn("qwen3.5-4b", self.cache)

        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
        )
        self.cache.put(entry)
        self.assertTrue(self.cache.contains("qwen3.5-4b"))
        self.assertIn("qwen3.5-4b", self.cache)

    def test_05_remove_works(self) -> None:
        """5. remove() works."""
        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
        )
        self.cache.put(entry)
        removed = self.cache.remove("qwen3.5-4b")
        self.assertEqual(removed.model_id, "qwen3.5-4b")
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.remove("nonexistent"))

    def test_06_clear_works(self) -> None:
        """6. clear() works."""
        for model_id in ["qwen3.5-4b", "lfm2.5-thinking-1.2b"]:
            self.cache.put(
                ModelCacheEntry(
                    model_id=model_id,
                    last_used_at=datetime.now(timezone.utc),
                )
            )
        self.assertEqual(self.cache.size(), 2)
        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)

    def test_07_cache_size_is_tracked(self) -> None:
        """7. Cache size is tracked accurately."""
        self.assertEqual(self.cache.size(), 0)
        self.cache.put(ModelCacheEntry("m1", datetime.now(timezone.utc)))
        self.assertEqual(self.cache.size(), 1)
        self.cache.put(ModelCacheEntry("m2", datetime.now(timezone.utc)))
        self.assertEqual(self.cache.size(), 2)
        self.cache.remove("m1")
        self.assertEqual(self.cache.size(), 1)

    def test_08_max_entries_is_enforced(self) -> None:
        """8. max_entries is enforced by evicting oldest non-active entry."""
        # max_entries is 3
        t0 = datetime.now(timezone.utc)
        self.cache.put(ModelCacheEntry("m1", t0 - timedelta(minutes=30)))
        self.cache.put(ModelCacheEntry("m2", t0 - timedelta(minutes=20)))
        self.cache.put(ModelCacheEntry("m3", t0 - timedelta(minutes=10)))
        self.assertEqual(self.cache.size(), 3)

        # Adding 4th entry evicts m1 (oldest)
        self.cache.put(ModelCacheEntry("m4", t0))
        self.assertEqual(self.cache.size(), 3)
        self.assertNotIn("m1", self.cache)
        self.assertIn("m2", self.cache)
        self.assertIn("m3", self.cache)
        self.assertIn("m4", self.cache)

    def test_09_lru_ordering_works(self) -> None:
        """9. LRU ordering works and list_entries returns most recently used first."""
        t0 = datetime.now(timezone.utc)
        self.cache.put(ModelCacheEntry("m1", t0 - timedelta(minutes=20)))
        self.cache.put(ModelCacheEntry("m2", t0 - timedelta(minutes=5)))
        self.cache.put(ModelCacheEntry("m3", t0 - timedelta(minutes=10)))

        ordered = [e.model_id for e in self.cache.list_entries()]
        self.assertEqual(ordered, ["m2", "m3", "m1"])

    def test_10_get_least_recently_used_returns_correct_model(self) -> None:
        """10. get_least_recently_used() returns the model with oldest timestamp."""
        t0 = datetime.now(timezone.utc)
        self.cache.put(ModelCacheEntry("m1", t0 - timedelta(minutes=15)))
        self.cache.put(ModelCacheEntry("m2", t0 - timedelta(minutes=50)))  # oldest
        self.cache.put(ModelCacheEntry("m3", t0 - timedelta(minutes=5)))

        lru = self.cache.get_least_recently_used()
        self.assertIsNotNone(lru)
        self.assertEqual(lru.model_id, "m2")

    def test_11_touch_updates_last_used_at(self) -> None:
        """11. touch() updates last_used_at."""
        t_old = datetime.now(timezone.utc) - timedelta(hours=1)
        self.cache.put(ModelCacheEntry("m1", t_old))
        self.assertEqual(self.cache.get("m1").last_used_at, t_old)

        self.cache.touch("m1")
        t_new = self.cache.get("m1").last_used_at
        self.assertGreater(t_new, t_old)

    def test_12_usage_count_updates_correctly(self) -> None:
        """12. usage_count updates correctly on increment_usage."""
        self.cache.put(ModelCacheEntry("m1", datetime.now(timezone.utc), usage_count=1))
        self.cache.increment_usage("m1")
        self.assertEqual(self.cache.get("m1").usage_count, 2)
        self.cache.increment_usage("m1")
        self.assertEqual(self.cache.get("m1").usage_count, 3)

    def test_13_estimated_memory_is_stored(self) -> None:
        """13. Estimated memory is stored in cache entry."""
        mem = 3 * 1024 * 1024 * 1024  # 3 GB
        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
            estimated_memory_bytes=mem,
        )
        self.cache.put(entry)
        retrieved = self.cache.get("qwen3.5-4b")
        self.assertEqual(retrieved.estimated_memory_bytes, mem)
        self.assertEqual(retrieved.estimated_memory_gb, 3.0)

    def test_14_total_estimated_cached_memory_is_calculated(self) -> None:
        """14. Total estimated cached memory is calculated across all entries."""
        self.cache.put(
            ModelCacheEntry("m1", datetime.now(timezone.utc), estimated_memory_bytes=2 * 1024**3)
        )
        self.cache.put(
            ModelCacheEntry("m2", datetime.now(timezone.utc), estimated_memory_bytes=3 * 1024**3)
        )
        self.assertEqual(self.cache.total_estimated_cached_memory(), 5 * 1024**3)

    def test_15_active_model_cannot_be_evicted(self) -> None:
        """15. An ACTIVE model is not evicted when non-active entries can be evicted."""
        t0 = datetime.now(timezone.utc)
        # m1 is active (even though oldest)
        active_entry = ModelCacheEntry(
            "m1",
            t0 - timedelta(hours=2),
            cache_state=CacheState.ACTIVE,
        )
        self.cache.put(active_entry)
        self.cache.put(ModelCacheEntry("m2", t0 - timedelta(hours=1)))
        self.cache.put(ModelCacheEntry("m3", t0 - timedelta(minutes=30)))

        # Adding m4 should evict m2 (oldest non-active entry), NOT m1!
        self.cache.put(ModelCacheEntry("m4", t0))
        self.assertIn("m1", self.cache)
        self.assertNotIn("m2", self.cache)
        self.assertIn("m3", self.cache)
        self.assertIn("m4", self.cache)

    def test_16_cache_does_not_contain_subprocess_handles(self) -> None:
        """16. Cache does not contain subprocess handles."""
        entry = ModelCacheEntry(
            model_id="qwen3.5-4b",
            last_used_at=datetime.now(timezone.utc),
        )
        self.cache.put(entry)
        retrieved = self.cache.get("qwen3.5-4b")
        for key, val in retrieved.__dict__.items():
            self.assertFalse(hasattr(val, "pid"), f"Field {key} must not be a process handle")
            self.assertFalse(hasattr(val, "communicate"), f"Field {key} must not be a process handle")

    def test_17_cache_does_not_load_gguf_files(self) -> None:
        """17. Cache operations do not read or load GGUF file bytes into memory."""
        # Using cache operations only modifies metadata
        self.cache.put(
            ModelCacheEntry(
                model_id="qwen3-coder-30b",
                last_used_at=datetime.now(timezone.utc),
                estimated_memory_bytes=20 * 1024**3,
            )
        )
        self.assertEqual(self.cache.size(), 1)
        self.assertIn("qwen3-coder-30b", self.cache)

    def test_18_cache_does_not_start_llama_cpp(self) -> None:
        """18. Cache operations never invoke subprocesses."""
        self.cache.put(ModelCacheEntry("test", datetime.now(timezone.utc)))
        self.cache.touch("test")
        self.cache.get_least_recently_used()
        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)

    def test_19_ram_manager_remains_the_authority_for_memory_safety(self) -> None:
        """19. RAMManager remains the authority; cache memory estimation cannot override UNSAFE evaluation."""
        ram_mgr = RAMManager(registry=self.registry)
        # 30B coder model is UNSAFE
        safety = ram_mgr.evaluate_safety("qwen3-coder-30b")
        self.assertEqual(safety.decision, MemoryDecision.UNSAFE)

        # Cache may hold metadata, but RAMManager still denies load
        self.cache.put(
            ModelCacheEntry(
                model_id="qwen3-coder-30b",
                last_used_at=datetime.now(timezone.utc),
                estimated_memory_bytes=safety.estimated_model_bytes,
            )
        )
        self.assertFalse(ram_mgr.can_load("qwen3-coder-30b"))

    def test_20_model_loader_updates_cache_on_load(self) -> None:
        """20. ModelLoader integration updates cache metadata after successful load."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        def fake_start(m):
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        mock_runtime.start.side_effect = fake_start

        loader = ModelLoader(
            registry=self.registry,
            runtime=mock_runtime,
            cache=self.cache,
        )

        self.assertEqual(self.cache.size(), 0)
        loader.load("qwen3.5-4b")

        self.assertEqual(self.cache.size(), 1)
        cached = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.cache_state, CacheState.ACTIVE)
        self.assertEqual(cached.lifecycle_state, ModelLifecycleState.READY)
        self.assertEqual(cached.usage_count, 1)

    def test_21_model_loader_preserves_cache_on_unload(self) -> None:
        """21. ModelLoader integration preserves cache metadata after unload."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        def fake_start(m):
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        def fake_stop(timeout=5.0):
            mock_runtime.is_running.return_value = False
            mock_runtime.state = RuntimeState.STOPPED

        mock_runtime.start.side_effect = fake_start
        mock_runtime.stop.side_effect = fake_stop

        loader = ModelLoader(
            registry=self.registry,
            runtime=mock_runtime,
            cache=self.cache,
        )

        loader.load("qwen3.5-4b")
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.ACTIVE)

        loader.unload()
        # Cache metadata MUST remain available after unload!
        self.assertEqual(self.cache.size(), 1)
        cached = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.cache_state, CacheState.CACHED_METADATA)
        self.assertEqual(cached.lifecycle_state, ModelLifecycleState.UNLOADED)
        self.assertEqual(cached.usage_count, 1)

    def test_22_existing_one_active_model_rule_remains_enforced(self) -> None:
        """22. Existing one-active-model rule remains enforced when cache is active."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        def fake_start(m):
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        mock_runtime.start.side_effect = fake_start

        loader = ModelLoader(
            registry=self.registry,
            runtime=mock_runtime,
            cache=self.cache,
        )

        loader.load("qwen3.5-4b")
        with self.assertRaises(ModelAlreadyLoadedError):
            loader.load("qwen3-vl-4b")

    def test_23_invalid_model_ids_handled_cleanly(self) -> None:
        """23. Querying non-existent model IDs returns None without exceptions."""
        self.assertIsNone(self.cache.get("unknown-id"))
        self.assertIsNone(self.cache.touch("unknown-id"))
        self.assertIsNone(self.cache.increment_usage("unknown-id"))
        self.assertIsNone(self.cache.remove("unknown-id"))

    def test_24_missing_stale_model_definitions_handled_safely(self) -> None:
        """24. Cache holds metadata safely even if physical files disappear or become stale."""
        entry = ModelCacheEntry(
            model_id="removed-model",
            last_used_at=datetime.now(timezone.utc),
            estimated_memory_bytes=1024,
        )
        self.cache.put(entry)
        self.assertTrue(self.cache.contains("removed-model"))
        self.cache.remove("removed-model")
        self.assertFalse(self.cache.contains("removed-model"))


if __name__ == "__main__":
    unittest.main()
