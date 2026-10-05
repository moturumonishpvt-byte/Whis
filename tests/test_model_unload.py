"""Tests for WHIS Stage 3.7: Unload Manager.

Validates deterministic model unloading, ModelLoader integration,
active model protection, lifecycle and cache state transitions,
LRU candidate identification, and memory pressure evaluations without
direct subprocess manipulation or automatic background execution.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import unittest
from unittest.mock import MagicMock, patch

from app.ai.cache import CacheState, ModelCache, ModelCacheEntry
from app.ai.lifecycle import ModelLifecycle, ModelLifecycleState
from app.ai.loader import ModelLoader
from app.ai.model_manager import ModelDefinition, ModelRegistry
from app.ai.ram_manager import (
    DEFAULT_SAFETY_RESERVE_BYTES,
    MemoryDecision,
    MemoryProvider,
    MemorySnapshot,
    RAMManager,
)
from app.ai.runtime import (
    DevicePolicy,
    RuntimeAdapter,
    RuntimeConfig,
    RuntimeState,
)
from app.ai.unload import (
    ModelUnloadError,
    UnloadAction,
    UnloadDecision,
    UnloadManager,
    UnloadResult,
)


class TestModelUnloadManager(unittest.TestCase):
    """Unit test suite for WHIS UnloadManager."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()

        # Mock runtime adapter
        self.mock_runtime = MagicMock(spec=RuntimeAdapter)
        self.mock_runtime.is_running.return_value = False
        self.mock_runtime.state = RuntimeState.STOPPED
        self.mock_runtime.config = RuntimeConfig(
            executable="llama-cli",
            device_policy=DevicePolicy.CPU_ONLY,
        )

        # Default mock start and stop behavior
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        def fake_stop(timeout: float = 5.0) -> None:
            self.mock_runtime.is_running.return_value = False
            self.mock_runtime.state = RuntimeState.STOPPED

        self.mock_runtime.start.side_effect = fake_start
        self.mock_runtime.stop.side_effect = fake_stop

        # Mock memory provider (16 GB total, 8 GB available)
        self.mock_mem_provider = MagicMock(spec=MemoryProvider)
        self.snapshot_16gb_total_8gb_avail = MemorySnapshot(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=8 * 1024 * 1024 * 1024,
            used_bytes=8 * 1024 * 1024 * 1024,
            percent_used=50.0,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_mem_provider.get_memory_snapshot.return_value = (
            self.snapshot_16gb_total_8gb_avail
        )
        self.mock_mem_provider.get_process_memory.return_value = None

        self.ram_manager = RAMManager(
            registry=self.registry,
            memory_provider=self.mock_mem_provider,
            safety_reserve_bytes=DEFAULT_SAFETY_RESERVE_BYTES,
        )

        self.lifecycle = ModelLifecycle()
        self.cache = ModelCache(max_entries=5)

        self.loader = ModelLoader(
            registry=self.registry,
            runtime=self.mock_runtime,
            lifecycle=self.lifecycle,
            ram_manager=self.ram_manager,
            cache=self.cache,
        )

        self.unload_manager = UnloadManager(
            loader=self.loader,
            ram_manager=self.ram_manager,
            cache=self.cache,
        )

    def _load_model(self, model_id: str = "qwen3.5-4b") -> None:
        """Helper to load a model through ModelLoader."""
        self.loader.load(model_id)
        self.assertTrue(self.loader.is_loaded(model_id))

    def test_01_empty_manager_initialization(self) -> None:
        """1. Empty manager initializes correctly with expected references and defaults."""
        self.assertIs(self.unload_manager.loader, self.loader)
        self.assertIs(self.unload_manager.ram_manager, self.ram_manager)
        self.assertIs(self.unload_manager.cache, self.cache)
        self.assertFalse(self.unload_manager.has_active_model)
        self.assertIsNone(self.unload_manager.active_model_id)
        self.assertEqual(
            self.unload_manager.active_lifecycle_state,
            ModelLifecycleState.UNLOADED,
        )

    def test_02_no_active_model(self) -> None:
        """2. When no model is active, properties and query methods reflect inactive state."""
        self.assertFalse(self.unload_manager.has_active_model)
        self.assertIsNone(self.unload_manager.active_model_id)
        self.assertFalse(self.unload_manager.can_unload())
        self.assertFalse(self.unload_manager.can_unload("qwen3.5-4b"))

    def test_03_unload_current_with_no_model(self) -> None:
        """3. unload_current() when no model is active returns safe non-success result."""
        result = self.unload_manager.unload_current()
        self.assertIsInstance(result, UnloadResult)
        self.assertFalse(result.success)
        self.assertIsNone(result.model_id)
        self.assertEqual(result.previous_state, ModelLifecycleState.UNLOADED)
        self.assertEqual(result.final_state, ModelLifecycleState.UNLOADED)
        self.assertIn("No active model", result.reason)
        self.assertFalse(self.mock_runtime.stop.called)

    def test_04_successful_unload_of_active_model(self) -> None:
        """4. Successful unload of currently active model returns structured success."""
        self._load_model("qwen3.5-4b")
        result = self.unload_manager.unload_current(reason="Maintenance")

        self.assertTrue(result.success)
        self.assertEqual(result.model_id, "qwen3.5-4b")
        self.assertEqual(result.previous_state, ModelLifecycleState.READY)
        self.assertEqual(result.final_state, ModelLifecycleState.UNLOADED)
        self.assertEqual(result.reason, "Maintenance")
        self.assertIsNone(result.error)

    def test_05_model_loader_is_actually_used(self) -> None:
        """5. UnloadManager delegates unloading strictly through ModelLoader.unload()."""
        self._load_model("qwen3.5-4b")
        with patch.object(self.loader, "unload", wraps=self.loader.unload) as spy_unload:
            result = self.unload_manager.unload_current()
            self.assertTrue(result.success)
            self.assertTrue(spy_unload.called)

    def test_06_runtime_reaches_stopped(self) -> None:
        """6. After unloading, runtime state reaches STOPPED and is_running is False."""
        self._load_model("qwen3.5-4b")
        self.assertEqual(self.mock_runtime.state, RuntimeState.RUNNING)

        self.unload_manager.unload_current()

        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)
        self.assertFalse(self.mock_runtime.is_running())

    def test_07_lifecycle_reaches_unloaded(self) -> None:
        """7. After unloading, ModelLifecycle state reaches UNLOADED."""
        self._load_model("qwen3.5-4b")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)

        self.unload_manager.unload_current()

        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertEqual(self.unload_manager.active_lifecycle_state, ModelLifecycleState.UNLOADED)

    def test_08_cache_changes_active_to_cached_metadata(self) -> None:
        """8. ModelCache entry state transitions from ACTIVE to CACHED_METADATA."""
        self._load_model("qwen3.5-4b")
        cached = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.cache_state, CacheState.ACTIVE)

        self.unload_manager.unload_current()

        cached_after = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(cached_after)
        self.assertEqual(cached_after.cache_state, CacheState.CACHED_METADATA)

    def test_09_cache_metadata_survives_unload(self) -> None:
        """9. Cache entry and its metadata dictionary remain in cache after unload."""
        self._load_model("qwen3.5-4b")
        self.unload_manager.unload_current()

        self.assertTrue(self.cache.contains("qwen3.5-4b"))
        entry = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.metadata.get("role"), "main_brain")

    def test_10_usage_count_survives_unload(self) -> None:
        """10. Cache usage_count is preserved across model unload."""
        self._load_model("qwen3.5-4b")
        entry_before = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry_before.usage_count, 1)

        self.unload_manager.unload_current()

        entry_after = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry_after.usage_count, 1)

    def test_11_last_used_at_survives_unload(self) -> None:
        """11. last_used_at timestamp is preserved in cache metadata after unload."""
        self._load_model("qwen3.5-4b")
        entry_before = self.cache.get("qwen3.5-4b")
        timestamp_before = entry_before.last_used_at

        self.unload_manager.unload_current()

        entry_after = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry_after.last_used_at, timestamp_before)
        self.assertIsNotNone(entry_after.last_used_at.tzinfo)

    def test_12_estimated_memory_survives_unload(self) -> None:
        """12. Estimated memory bytes are preserved in cache entry after unload."""
        self._load_model("qwen3.5-4b")
        entry_before = self.cache.get("qwen3.5-4b")
        est_bytes_before = entry_before.estimated_memory_bytes
        self.assertGreater(est_bytes_before, 0)

        self.unload_manager.unload_current()

        entry_after = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry_after.estimated_memory_bytes, est_bytes_before)

    def test_13_unload_model_active_model(self) -> None:
        """13. unload_model(active_model) successfully unloads the targeted active model."""
        self._load_model("qwen3.5-4b")

        result = self.unload_manager.unload_model("qwen3.5-4b")

        self.assertTrue(result.success)
        self.assertEqual(result.model_id, "qwen3.5-4b")
        self.assertFalse(self.loader.is_loaded())

    def test_14_unload_model_non_active_model(self) -> None:
        """14. unload_model(non_active_model) does NOT unload active model (protection)."""
        self._load_model("qwen3.5-4b")

        result = self.unload_manager.unload_model("lfm2.5-thinking-1.2b")

        self.assertFalse(result.success)
        self.assertEqual(result.model_id, "lfm2.5-thinking-1.2b")
        self.assertIn("Active model is protected", result.reason)
        # Active model is still loaded and running
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))
        self.assertTrue(self.mock_runtime.is_running())

    def test_15_cannot_unload_nonexistent_model(self) -> None:
        """15. Calling unload_model for an unregistered model returns safe failure without crashing."""
        self._load_model("qwen3.5-4b")

        result = self.unload_manager.unload_model("nonexistent-fake-model-xyz")

        self.assertFalse(result.success)
        self.assertEqual(result.model_id, "nonexistent-fake-model-xyz")
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))

    def test_16_can_unload_active_model(self) -> None:
        """16. can_unload() returns True for currently loaded and ready active model."""
        self._load_model("qwen3.5-4b")

        self.assertTrue(self.unload_manager.can_unload())
        self.assertTrue(self.unload_manager.can_unload("qwen3.5-4b"))

    def test_17_can_unload_inactive_model(self) -> None:
        """17. can_unload() returns False for any inactive model ID or when unloaded."""
        self.assertFalse(self.unload_manager.can_unload())
        self.assertFalse(self.unload_manager.can_unload("qwen3.5-4b"))

        self._load_model("qwen3.5-4b")
        self.assertFalse(self.unload_manager.can_unload("lfm2.5-thinking-1.2b"))

    def test_18_get_unload_candidate_uses_model_cache_lru(self) -> None:
        """18. get_unload_candidate() selects candidate with oldest last_used_at."""
        t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 10, 30, 0, tzinfo=timezone.utc)

        entry1 = ModelCacheEntry(
            model_id="model-a",
            last_used_at=t1,
            cache_state=CacheState.CACHED_METADATA,
        )
        entry2 = ModelCacheEntry(
            model_id="model-b",
            last_used_at=t2,
            cache_state=CacheState.CACHED_METADATA,
        )
        self.cache.put(entry1)
        self.cache.put(entry2)

        candidate = self.unload_manager.get_unload_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.model_id, "model-a")

    def test_19_active_model_is_protected(self) -> None:
        """19. get_unload_candidate(exclude_active=True) protects the active model."""
        now = datetime.now(timezone.utc)
        older = now - timedelta(hours=1)

        # Inactive older entry
        entry_inactive = ModelCacheEntry(
            model_id="inactive-model",
            last_used_at=older,
            cache_state=CacheState.CACHED_METADATA,
        )
        self.cache.put(entry_inactive)

        # Active newer entry
        self._load_model("qwen3.5-4b")

        candidate = self.unload_manager.get_unload_candidate(exclude_active=True)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.model_id, "inactive-model")

        # When only active model exists in cache
        self.cache.remove("inactive-model")
        candidate_none = self.unload_manager.get_unload_candidate(exclude_active=True)
        self.assertIsNone(candidate_none)

    def test_20_memory_pressure_decision(self) -> None:
        """20. unload_if_needed() detects memory pressure and returns UNLOAD_REQUIRED."""
        self._load_model("qwen3.5-4b")

        # Simulate low available RAM (1.0 GB available < 2.0 GB reserve)
        low_memory_snapshot = MemorySnapshot(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=1 * 1024 * 1024 * 1024,
            used_bytes=15 * 1024 * 1024 * 1024,
            percent_used=93.75,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_mem_provider.get_memory_snapshot.return_value = low_memory_snapshot

        decision = self.unload_manager.unload_if_needed()

        self.assertIsInstance(decision, UnloadDecision)
        self.assertEqual(decision.action, UnloadAction.UNLOAD_REQUIRED)
        self.assertTrue(decision.should_unload)
        self.assertEqual(decision.model_id, "qwen3.5-4b")
        self.assertEqual(decision.memory_decision, MemoryDecision.UNSAFE)
        self.assertIn("RAM pressure detected", decision.reason)

    def test_21_unsafe_memory_condition(self) -> None:
        """21. Target model exceeding total machine safe capacity yields UNLOAD_NOT_POSSIBLE."""
        self._load_model("qwen3.5-4b")

        # qwen3-coder-30b requires ~19.4 GB, which exceeds 16 GB - 2 GB reserve = 14 GB safe RAM
        decision = self.unload_manager.unload_if_needed(target_model_id="qwen3-coder-30b")

        self.assertEqual(decision.action, UnloadAction.UNLOAD_NOT_POSSIBLE)
        self.assertFalse(decision.should_unload)
        self.assertEqual(decision.memory_decision, MemoryDecision.UNSAFE)
        self.assertIn("exceeds total system safe RAM", decision.reason)

    def test_22_unload_failure_handling(self) -> None:
        """22. Failure during runtime.stop() is captured and safely reported without crashing."""
        self._load_model("qwen3.5-4b")

        self.mock_runtime.stop.side_effect = RuntimeError("Subprocess termination timeout")

        result = self.unload_manager.unload_current()

        self.assertFalse(result.success)
        self.assertEqual(result.model_id, "qwen3.5-4b")
        self.assertIsNotNone(result.error)
        self.assertIn("Subprocess termination timeout", result.error)
        self.assertEqual(result.final_state, ModelLifecycleState.FAILED)

    def test_23_invalid_lifecycle_state_handling(self) -> None:
        """23. Models in non-unloadable lifecycle states (BUSY) reject unload safely."""
        self._load_model("qwen3.5-4b")

        # Simulate busy inference state
        self.lifecycle.begin_use()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.BUSY)

        self.assertFalse(self.unload_manager.can_unload())

        result = self.unload_manager.unload_current()
        self.assertFalse(result.success)
        self.assertIn("cannot be unloaded while in 'BUSY' state", result.reason)

        # Reset state to READY and verify unload proceeds
        self.lifecycle.end_use()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertTrue(self.unload_manager.can_unload())

    def test_24_no_subprocess_direct_control(self) -> None:
        """24. UnloadManager does not import subprocess or contain direct OS process hooks."""
        unload_file = Path(__file__).resolve().parent.parent / "app" / "ai" / "unload.py"
        with open(unload_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(unload_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name, "subprocess")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "subprocess")

        # Verify no direct subprocess handle stored in instance
        self.assertFalse(hasattr(self.unload_manager, "process"))
        self.assertFalse(hasattr(self.unload_manager, "_process"))

    def test_25_no_automatic_background_unloading(self) -> None:
        """25. UnloadManager does not start background threads, timers, or periodic watchers."""
        thread_count_before = threading.active_count()

        mgr = UnloadManager(loader=self.loader)
        _ = mgr.unload_if_needed()
        _ = mgr.can_unload()

        thread_count_after = threading.active_count()
        self.assertEqual(thread_count_before, thread_count_after)
        self.assertFalse(hasattr(mgr, "_timer"))
        self.assertFalse(hasattr(mgr, "_thread"))
        self.assertFalse(hasattr(mgr, "_watcher"))


if __name__ == "__main__":
    unittest.main()
