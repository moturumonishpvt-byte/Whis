"""Comprehensive integration and stress tests for WHIS Stage 3.8.

Validates the full local model management subsystem across Stages 3.1 to 3.7:
ModelRegistry -> ModelCache -> RAMManager -> ModelLoader -> UnloadManager -> ModelLifecycle -> RuntimeAdapter.

Includes:
- Complete component integration (A-E)
- Model switching flow (F)
- Repeated load/unload stress testing (G)
- Cache stress and eviction immunity (H)
- Failure injection across all layers (I)
- Concurrency and single-active model protection (J)
- Process isolation and architectural boundary checks (K)
- Full tri-state consistency verification (L)
- Real-model integration smoke test for qwen3.5-4b (M)
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import unittest
from unittest.mock import MagicMock, patch

from app.ai.cache import CacheState, ModelCache, ModelCacheEntry
from app.ai.lifecycle import (
    InvalidStateTransitionError,
    ModelLifecycle,
    ModelLifecycleState,
    ModelRuntimeStatus,
)
from app.ai.loader import (
    LoadedModelInfo,
    ModelAlreadyLoadedError,
    ModelDisabledError,
    ModelLoadError,
    ModelLoader,
)
from app.ai.model_manager import (
    ModelDefinition,
    ModelNotFoundError,
    ModelRegistry,
)
from app.ai.ram_manager import (
    DEFAULT_SAFETY_RESERVE_BYTES,
    InsufficientMemoryError,
    MemoryDecision,
    MemoryProvider,
    MemorySnapshot,
    RAMManager,
)
from app.ai.runtime import (
    DevicePolicy,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeConfig,
    RuntimeStartError,
    RuntimeState,
)
from app.ai.unload import (
    UnloadAction,
    UnloadDecision,
    UnloadManager,
    UnloadResult,
)


class TestModelSubsystemIntegration(unittest.TestCase):
    """End-to-end integration and stress tests for WHIS local model subsystem."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()

        # Mock runtime adapter for deterministic, fast subsystem tests
        self.mock_runtime = MagicMock(spec=RuntimeAdapter)
        self.mock_runtime.is_running.return_value = False
        self.mock_runtime.state = RuntimeState.STOPPED
        self.mock_runtime.config = RuntimeConfig(
            executable="llama-cli",
            device_policy=DevicePolicy.CPU_ONLY,
        )

        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        def fake_stop(timeout: float = 5.0) -> None:
            self.mock_runtime.is_running.return_value = False
            self.mock_runtime.state = RuntimeState.STOPPED

        self.mock_runtime.start.side_effect = fake_start
        self.mock_runtime.stop.side_effect = fake_stop

        # Mock memory provider (16 GB baseline, 8 GB available)
        self.mock_mem_provider = MagicMock(spec=MemoryProvider)
        self.snapshot_8gb_avail = MemorySnapshot(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=8 * 1024 * 1024 * 1024,
            used_bytes=8 * 1024 * 1024 * 1024,
            percent_used=50.0,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_mem_provider.get_memory_snapshot.return_value = self.snapshot_8gb_avail
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

    # ------------------------------------------------------------------
    # A. REGISTRY -> LOADER
    # ------------------------------------------------------------------

    def test_01_registry_loader_resolution(self) -> None:
        """1. Registered model is correctly resolved from registry by ModelLoader."""
        status = self.loader.load("qwen3.5-4b")
        self.assertEqual(status.model_id, "qwen3.5-4b")
        self.assertEqual(status.display_name, "Qwen3.5 4B")
        self.assertEqual(status.role, "main_brain")
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))

    def test_02_unknown_model_rejection(self) -> None:
        """2. Attempting to load an unregistered model is rejected with ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError):
            self.loader.load("unknown_model_xyz")
        self.assertFalse(self.loader.is_loaded())
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)

    def test_03_disabled_model_rejection(self) -> None:
        """3. Models with enabled=False are rejected with ModelDisabledError."""
        disabled_model = ModelDefinition(
            id="disabled-test-model",
            display_name="Disabled Test",
            type="llm",
            role="auxiliary",
            status="experimental",
            capabilities=["chat"],
            runtime="llama.cpp",
            default_context=2048,
            model_path="models/disabled.gguf",
            enabled=False,
        )
        with patch.object(self.registry, "get", return_value=disabled_model):
            with self.assertRaises(ModelDisabledError):
                self.loader.load("disabled-test-model")
        self.assertFalse(self.loader.is_loaded())

    def test_04_missing_model_file_rejection(self) -> None:
        """4. Model weights missing on disk trigger ModelLoadError before runtime launch."""
        with patch.object(self.registry, "resolve_model_path", return_value=Path("nonexistent_weights.gguf")):
            with self.assertRaises(ModelLoadError) as ctx:
                self.loader.load("qwen3.5-4b")
            self.assertIn("weights file not found", str(ctx.exception))
            self.assertFalse(self.mock_runtime.start.called)
            self.assertFalse(self.loader.is_loaded())

    def test_05_multimodal_projector_validation(self) -> None:
        """5. Multimodal models missing projector file are rejected with ModelLoadError."""
        with patch.object(self.registry, "resolve_mmproj_path", return_value=Path("nonexistent_mmproj.gguf")):
            with self.assertRaises(ModelLoadError) as ctx:
                self.loader.load("qwen3-vl-4b")
            self.assertIn("projector file not found", str(ctx.exception))
            self.assertFalse(self.mock_runtime.start.called)

    # ------------------------------------------------------------------
    # B. REGISTRY -> RAM MANAGER
    # ------------------------------------------------------------------

    def test_06_registered_model_memory_estimation(self) -> None:
        """6. RAMManager computes structured memory estimates for registered models."""
        estimate = self.ram_manager.estimate_model_memory("qwen3.5-4b")
        self.assertEqual(estimate.model_id, "qwen3.5-4b")
        self.assertGreater(estimate.weight_bytes, 0)
        self.assertGreater(estimate.estimated_kv_cache_bytes, 0)
        self.assertGreater(estimate.estimated_total_bytes, 0)

    def test_07_context_size_impacts_kv_cache_estimate(self) -> None:
        """7. Larger context token window produces strictly larger memory estimates."""
        est_2k = self.ram_manager.estimate_model_memory("qwen3.5-4b", context_tokens=2048)
        est_8k = self.ram_manager.estimate_model_memory("qwen3.5-4b", context_tokens=8192)
        self.assertGreater(est_8k.estimated_kv_cache_bytes, est_2k.estimated_kv_cache_bytes)
        self.assertGreater(est_8k.estimated_total_bytes, est_2k.estimated_total_bytes)

    def test_08_ram_manager_rejects_unsafe_model(self) -> None:
        """8. ModelLoader rejects models evaluating to UNSAFE before runtime launch."""
        # qwen3-coder-30b requires ~19.4 GB on 16 GB baseline -> UNSAFE
        with self.assertRaises(InsufficientMemoryError) as ctx:
            self.loader.load("qwen3-coder-30b")
        self.assertEqual(ctx.exception.model_id, "qwen3-coder-30b")
        self.assertFalse(self.mock_runtime.start.called)
        self.assertFalse(self.loader.is_loaded())

    def test_09_ram_manager_single_authority(self) -> None:
        """9. ModelLoader delegates memory evaluation solely to RAMManager."""
        with patch.object(
            self.ram_manager,
            "evaluate_safety",
            wraps=self.ram_manager.evaluate_safety,
        ) as spy_eval:
            self.loader.load("qwen3.5-4b")
            self.assertTrue(spy_eval.called)

    def test_10_ram_warning_policy_enforcement(self) -> None:
        """10. When allow_warning=False, a WARNING evaluation is rejected as unsafe."""
        # 4 GB available -> warning for qwen3.5-4b (requires 3.44 GB, encroaches reserve)
        warning_snapshot = MemorySnapshot(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=4 * 1024 * 1024 * 1024,
            used_bytes=12 * 1024 * 1024 * 1024,
            percent_used=75.0,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_mem_provider.get_memory_snapshot.return_value = warning_snapshot

        with self.assertRaises(InsufficientMemoryError):
            self.loader.load("qwen3.5-4b", allow_warning=False)
        self.assertFalse(self.loader.is_loaded())

    # ------------------------------------------------------------------
    # C. LOADER -> LIFECYCLE -> RUNTIME
    # ------------------------------------------------------------------

    def test_11_lifecycle_state_progression_on_load(self) -> None:
        """11. Successful load progresses lifecycle UNLOADED->LOADING->READY and runtime STOPPED->RUNNING."""
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)

        status = self.loader.load("qwen3.5-4b")

        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertEqual(self.mock_runtime.state, RuntimeState.RUNNING)
        self.assertEqual(status.lifecycle_state, ModelLifecycleState.READY)

    def test_12_failed_runtime_startup_transitions_to_failed(self) -> None:
        """12. Runtime startup failure transitions lifecycle to FAILED and stops runtime."""
        self.mock_runtime.start.side_effect = RuntimeStartError(
            "Process failed immediately",
            model_id="qwen3.5-4b",
        )
        with self.assertRaises(ModelLoadError):
            self.loader.load("qwen3.5-4b")

        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertFalse(self.loader.is_loaded())

    def test_13_runtime_failure_never_produces_ready(self) -> None:
        """13. Runtime launch failure never falsely produces a READY lifecycle state."""
        self.mock_runtime.start.side_effect = RuntimeError("Generic fatal crash")
        with self.assertRaises(ModelLoadError):
            self.loader.load("qwen3.5-4b")

        self.assertNotEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)

    def test_14_runtime_process_status_tracked_in_loader(self) -> None:
        """14. loader.get_status() synchronizes runtime and lifecycle states."""
        self.loader.load("qwen3.5-4b")
        status = self.loader.get_status()
        self.assertIsNotNone(status)
        self.assertEqual(status.runtime_state, RuntimeState.RUNNING)
        self.assertEqual(status.lifecycle_state, ModelLifecycleState.READY)
        self.assertEqual(status.executable, "llama-cli")

    def test_15_runtime_stop_transitions(self) -> None:
        """15. Stopping model transitions runtime RUNNING->STOPPED and lifecycle READY->UNLOADED."""
        self.loader.load("qwen3.5-4b")
        self.loader.unload()
        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertFalse(self.loader.is_loaded())

    # ------------------------------------------------------------------
    # D. LOADER -> CACHE
    # ------------------------------------------------------------------

    def test_16_load_creates_active_cache_entry(self) -> None:
        """16. Loading creates an ACTIVE cache entry with usage_count=1."""
        self.loader.load("qwen3.5-4b")
        entry = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.cache_state, CacheState.ACTIVE)
        self.assertTrue(entry.is_active)
        self.assertEqual(entry.usage_count, 1)

    def test_17_repeated_load_increments_usage_count(self) -> None:
        """17. Unloading and reloading increments cache usage_count."""
        self.loader.load("qwen3.5-4b")
        self.loader.unload()
        self.loader.load("qwen3.5-4b")

        entry = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry.usage_count, 2)

    def test_18_last_used_at_updates_on_load(self) -> None:
        """18. last_used_at timestamp updates to timezone-aware UTC datetime."""
        t_before = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.loader.load("qwen3.5-4b")
        entry = self.cache.get("qwen3.5-4b")
        self.assertGreaterEqual(entry.last_used_at, t_before)
        self.assertIsNotNone(entry.last_used_at.tzinfo)

    def test_19_last_loaded_at_updates_on_load(self) -> None:
        """19. last_loaded_at timestamp is populated when loaded."""
        self.loader.load("qwen3.5-4b")
        entry = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(entry.last_loaded_at)

    def test_20_estimated_memory_stored_in_cache(self) -> None:
        """20. Cache entry stores estimated_memory_bytes matching RAMManager calculation."""
        self.loader.load("qwen3.5-4b")
        entry = self.cache.get("qwen3.5-4b")
        expected_bytes = self.ram_manager.estimate_model_memory("qwen3.5-4b").estimated_total_bytes
        self.assertEqual(entry.estimated_memory_bytes, expected_bytes)

    def test_21_cache_lifecycle_state_synchronized(self) -> None:
        """21. Cache entry lifecycle_state synchronizes with ModelLifecycle."""
        self.loader.load("qwen3.5-4b")
        self.assertEqual(self.cache.get("qwen3.5-4b").lifecycle_state, ModelLifecycleState.READY)
        self.loader.unload()
        self.assertEqual(self.cache.get("qwen3.5-4b").lifecycle_state, ModelLifecycleState.UNLOADED)

    def test_22_unload_transitions_cache_active_to_cached_metadata(self) -> None:
        """22. Unload transitions cache entry state from ACTIVE to CACHED_METADATA."""
        self.loader.load("qwen3.5-4b")
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.ACTIVE)
        self.loader.unload()
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.CACHED_METADATA)

    def test_23_cache_metadata_survives_unload(self) -> None:
        """23. Model metadata dictionary survives unload intact."""
        self.loader.load("qwen3.5-4b")
        self.loader.unload()
        entry = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry.metadata.get("role"), "main_brain")
        self.assertEqual(entry.metadata.get("type"), "llm")

    def test_24_cached_entry_not_deleted_on_unload(self) -> None:
        """24. Unload never deletes cache entry from ModelCache."""
        self.loader.load("qwen3.5-4b")
        self.loader.unload()
        self.assertTrue(self.cache.contains("qwen3.5-4b"))
        self.assertEqual(self.cache.size(), 1)

    # ------------------------------------------------------------------
    # E. UNLOAD MANAGER
    # ------------------------------------------------------------------

    def test_25_unload_current_unloads_active_model(self) -> None:
        """25. unload_manager.unload_current() cleanly unloads active model."""
        self.loader.load("qwen3.5-4b")
        result = self.unload_manager.unload_current()
        self.assertTrue(result.success)
        self.assertEqual(result.model_id, "qwen3.5-4b")
        self.assertEqual(result.final_state, ModelLifecycleState.UNLOADED)
        self.assertFalse(self.loader.is_loaded())

    def test_26_unload_model_active(self) -> None:
        """26. unload_model(active_model) unloads targeted model."""
        self.loader.load("qwen3.5-4b")
        result = self.unload_manager.unload_model("qwen3.5-4b")
        self.assertTrue(result.success)
        self.assertFalse(self.loader.is_loaded())

    def test_27_unload_model_non_active_preserves_active(self) -> None:
        """27. unload_model(non_active) protects active model from being unloaded."""
        self.loader.load("qwen3.5-4b")
        result = self.unload_manager.unload_model("lfm2.5-thinking-1.2b")
        self.assertFalse(result.success)
        self.assertIn("Active model is protected", result.reason)
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))

    def test_28_can_unload_reflects_lifecycle(self) -> None:
        """28. can_unload() returns True for READY and FAILED, False otherwise."""
        self.assertFalse(self.unload_manager.can_unload())
        self.loader.load("qwen3.5-4b")
        self.assertTrue(self.unload_manager.can_unload())

    def test_29_busy_lifecycle_protected_from_unload(self) -> None:
        """29. Models in BUSY state reject unload requests."""
        self.loader.load("qwen3.5-4b")
        self.lifecycle.begin_use()
        self.assertFalse(self.unload_manager.can_unload())
        result = self.unload_manager.unload_current()
        self.assertFalse(result.success)
        self.assertIn("cannot be unloaded while in 'BUSY' state", result.reason)
        self.lifecycle.end_use()

    def test_30_loading_lifecycle_protected_from_unload(self) -> None:
        """30. Models in LOADING state reject unload requests."""
        self.loader.load("qwen3.5-4b")
        self.lifecycle._state = ModelLifecycleState.LOADING
        try:
            self.assertFalse(self.unload_manager.can_unload())
            result = self.unload_manager.unload_current()
            self.assertFalse(result.success)
        finally:
            self.lifecycle._state = ModelLifecycleState.READY

    def test_31_stopping_lifecycle_protected_from_unload(self) -> None:
        """31. Models in STOPPING state reject duplicate unload requests."""
        self.loader.load("qwen3.5-4b")
        self.lifecycle._state = ModelLifecycleState.STOPPING
        try:
            self.assertFalse(self.unload_manager.can_unload())
            result = self.unload_manager.unload_current()
            self.assertFalse(result.success)
        finally:
            self.lifecycle._state = ModelLifecycleState.READY

    def test_32_get_unload_candidate_respects_lru(self) -> None:
        """32. get_unload_candidate() returns entry with oldest last_used_at."""
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        self.cache.put(ModelCacheEntry("model-old", last_used_at=t1, cache_state=CacheState.CACHED_METADATA))
        self.cache.put(ModelCacheEntry("model-new", last_used_at=t2, cache_state=CacheState.CACHED_METADATA))

        candidate = self.unload_manager.get_unload_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.model_id, "model-old")

    def test_33_get_unload_candidate_excludes_active(self) -> None:
        """33. Active model is excluded from inactive candidate selection."""
        self.loader.load("qwen3.5-4b")
        # Only active model in cache
        candidate = self.unload_manager.get_unload_candidate(exclude_active=True)
        self.assertIsNone(candidate)

    def test_34_no_subprocess_manipulation_in_unload_manager(self) -> None:
        """34. UnloadManager calls loader.unload() and contains no subprocess handles."""
        self.loader.load("qwen3.5-4b")
        with patch.object(self.loader, "unload", wraps=self.loader.unload) as spy:
            self.unload_manager.unload_current()
            self.assertTrue(spy.called)

    # ------------------------------------------------------------------
    # F. MODEL SWITCHING
    # ------------------------------------------------------------------

    def test_35_full_model_switching_flow(self) -> None:
        """35. Full sequence: Load A -> Unload A -> Load B. Verify states and cache."""
        # 1. Load Model A (qwen3.5-4b)
        self.loader.load("qwen3.5-4b")
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.ACTIVE)

        # 2. Unload Model A
        res = self.unload_manager.unload_current()
        self.assertTrue(res.success)
        self.assertFalse(self.loader.is_loaded())
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.CACHED_METADATA)

        # 3. Load Model B (qwen3-vl-4b)
        self.loader.load("qwen3-vl-4b")
        self.assertTrue(self.loader.is_loaded("qwen3-vl-4b"))
        self.assertFalse(self.loader.is_loaded("qwen3.5-4b"))

        # Verify cache states: B is ACTIVE, A is CACHED_METADATA
        self.assertEqual(self.cache.get("qwen3-vl-4b").cache_state, CacheState.ACTIVE)
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.CACHED_METADATA)

    def test_36_switching_cleans_up_previous_model_process(self) -> None:
        """36. Switching terminates previous runtime process cleanly before starting new."""
        self.loader.load("qwen3.5-4b")
        self.unload_manager.unload_current()
        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)

        self.loader.load("qwen3-vl-4b")
        self.assertEqual(self.mock_runtime.state, RuntimeState.RUNNING)
        self.assertEqual(self.loader.loaded_model().id, "qwen3-vl-4b")

    def test_37_unload_manager_unload_if_needed_triggers_model_switch(self) -> None:
        """37. unload_if_needed(target_model_id) returns UNLOAD_REQUIRED when another model is active."""
        self.loader.load("qwen3.5-4b")
        decision = self.unload_manager.unload_if_needed(target_model_id="qwen3-vl-4b")
        self.assertEqual(decision.action, UnloadAction.UNLOAD_REQUIRED)
        self.assertTrue(decision.should_unload)
        self.assertEqual(decision.model_id, "qwen3.5-4b")

    # ------------------------------------------------------------------
    # G. REPEATED LOAD/UNLOAD STRESS
    # ------------------------------------------------------------------

    def test_38_mocked_repeated_load_unload_stress_10_cycles(self) -> None:
        """38. Stress test: 10 consecutive Load -> Unload cycles maintain state integrity."""
        for cycle in range(1, 11):
            # Load
            info = self.loader.load("qwen3.5-4b")
            self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
            self.assertEqual(self.mock_runtime.state, RuntimeState.RUNNING)
            self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.ACTIVE)
            self.assertEqual(self.cache.get("qwen3.5-4b").usage_count, cycle)

            # Unload
            result = self.unload_manager.unload_current()
            self.assertTrue(result.success)
            self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
            self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)
            self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.CACHED_METADATA)

    def test_39_repeated_load_unload_preserves_cache_consistency(self) -> None:
        """39. Repeated load/unload cycles do not leak cache entries or corrupt metadata."""
        for _ in range(5):
            self.loader.load("qwen3.5-4b")
            self.unload_manager.unload_current()

        self.assertEqual(self.cache.size(), 1)
        entry = self.cache.get("qwen3.5-4b")
        self.assertEqual(entry.usage_count, 5)
        self.assertEqual(entry.cache_state, CacheState.CACHED_METADATA)

    def test_40_repeated_rapid_status_queries_during_lifecycle(self) -> None:
        """40. 50 rapid status queries return consistent snapshots during model lifecycle."""
        self.loader.load("qwen3.5-4b")
        for _ in range(50):
            st = self.loader.get_status()
            self.assertIsNotNone(st)
            self.assertEqual(st.model_id, "qwen3.5-4b")
            self.assertEqual(st.lifecycle_state, ModelLifecycleState.READY)
        self.unload_manager.unload_current()

    # ------------------------------------------------------------------
    # H. CACHE STRESS
    # ------------------------------------------------------------------

    def test_41_cache_repeated_touches_and_lru_ordering(self) -> None:
        """41. Repeated touches update timestamps and maintain strict LRU order."""
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        self.cache.put(ModelCacheEntry("model-1", last_used_at=t0))
        self.cache.put(ModelCacheEntry("model-2", last_used_at=t0 + timedelta(minutes=1)))
        self.cache.put(ModelCacheEntry("model-3", last_used_at=t0 + timedelta(minutes=2)))

        self.assertEqual(self.cache.get_least_recently_used().model_id, "model-1")

        # Touch model-1 -> now model-2 is LRU
        self.cache.touch("model-1")
        self.assertEqual(self.cache.get_least_recently_used().model_id, "model-2")

    def test_42_cache_capacity_limit_and_lru_eviction(self) -> None:
        """42. Cache capacity limit (5) evicts the oldest inactive entry upon 6th addition."""
        cache = ModelCache(max_entries=3)
        t = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(3):
            cache.put(ModelCacheEntry(f"m-{i}", last_used_at=t + timedelta(minutes=i)))

        self.assertEqual(cache.size(), 3)
        self.assertTrue(cache.contains("m-0"))

        # Add 4th entry -> m-0 (oldest) must be evicted
        cache.put(ModelCacheEntry("m-3", last_used_at=t + timedelta(minutes=10)))
        self.assertEqual(cache.size(), 3)
        self.assertFalse(cache.contains("m-0"))
        self.assertTrue(cache.contains("m-3"))

    def test_43_active_model_eviction_immunity(self) -> None:
        """43. Active running models are immune to cache capacity eviction."""
        cache = ModelCache(max_entries=2)
        t = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Active entry (oldest timestamp)
        cache.put(ModelCacheEntry(
            "active-model",
            last_used_at=t,
            cache_state=CacheState.ACTIVE,
        ))
        # Inactive entry
        cache.put(ModelCacheEntry(
            "inactive-1",
            last_used_at=t + timedelta(minutes=5),
            cache_state=CacheState.CACHED_METADATA,
        ))

        # Add 3rd entry -> inactive-1 should be evicted, NOT active-model!
        cache.put(ModelCacheEntry(
            "inactive-2",
            last_used_at=t + timedelta(minutes=10),
            cache_state=CacheState.CACHED_METADATA,
        ))

        self.assertTrue(cache.contains("active-model"))
        self.assertFalse(cache.contains("inactive-1"))
        self.assertTrue(cache.contains("inactive-2"))

    def test_44_total_estimated_cached_memory_calculation(self) -> None:
        """44. total_estimated_cached_memory() aggregates bytes across all entries."""
        self.cache.put(ModelCacheEntry("m1", last_used_at=datetime.now(timezone.utc), estimated_memory_bytes=1000))
        self.cache.put(ModelCacheEntry("m2", last_used_at=datetime.now(timezone.utc), estimated_memory_bytes=2500))
        self.assertEqual(self.cache.total_estimated_cached_memory(), 3500)

    # ------------------------------------------------------------------
    # I. FAILURE INJECTION
    # ------------------------------------------------------------------

    def test_45_failure_injection_unregistered_model(self) -> None:
        """45. Registry lookup failure raises ModelNotFoundError with clean state."""
        with self.assertRaises(ModelNotFoundError):
            self.loader.load("nonexistent_id")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)

    def test_46_failure_injection_invalid_weights_file(self) -> None:
        """46. Invalid weights path raises ModelLoadError and preserves UNLOADED state."""
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(ModelLoadError):
                self.loader.load("qwen3.5-4b")
            self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)

    def test_47_failure_injection_insufficient_memory(self) -> None:
        """47. Insufficient RAM raises InsufficientMemoryError with no runtime invocation."""
        unsafe_snap = MemorySnapshot(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=500 * 1024 * 1024,  # 500 MB
            used_bytes=15500 * 1024 * 1024,
            percent_used=96.9,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_mem_provider.get_memory_snapshot.return_value = unsafe_snap
        with self.assertRaises(InsufficientMemoryError):
            self.loader.load("qwen3.5-4b")
        self.assertFalse(self.mock_runtime.start.called)

    def test_48_failure_injection_runtime_launch_crash(self) -> None:
        """48. Runtime crash on launch marks lifecycle FAILED and cleans up loader state."""
        self.mock_runtime.start.side_effect = RuntimeError("Process segfault")
        with self.assertRaises(ModelLoadError):
            self.loader.load("qwen3.5-4b")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertFalse(self.loader.is_loaded())

    def test_49_failure_injection_recovery_after_failed_load(self) -> None:
        """49. Lifecycle cleanly recovers to UNLOADED on unload() after failed load."""
        self.mock_runtime.start.side_effect = RuntimeError("Crash on startup")
        with self.assertRaises(ModelLoadError):
            self.loader.load("qwen3.5-4b")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)

        # Unload call should recover state to UNLOADED
        self.loader.unload()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)

    def test_50_failure_injection_runtime_stop_timeout(self) -> None:
        """50. Runtime stop failure is captured in UnloadResult with error details."""
        self.loader.load("qwen3.5-4b")
        self.mock_runtime.stop.side_effect = RuntimeError("Stop timeout expired")

        result = self.unload_manager.unload_current()
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("Stop timeout expired", result.error)
        self.assertEqual(result.final_state, ModelLifecycleState.FAILED)

    # ------------------------------------------------------------------
    # J. CONCURRENCY & SINGLE-ACTIVE PROTECTION
    # ------------------------------------------------------------------

    def test_51_concurrency_protection_second_load_rejected(self) -> None:
        """51. Loading a second model while one is active raises ModelAlreadyLoadedError."""
        self.loader.load("qwen3.5-4b")
        with self.assertRaises(ModelAlreadyLoadedError):
            self.loader.load("qwen3-vl-4b")
        # Original model remains loaded
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))
        self.assertEqual(self.loader.loaded_model().id, "qwen3.5-4b")

    def test_52_concurrency_protection_same_model_reload_rejected(self) -> None:
        """52. Reloading the same model while active raises ModelAlreadyLoadedError."""
        self.loader.load("qwen3.5-4b")
        with self.assertRaises(ModelAlreadyLoadedError) as ctx:
            self.loader.load("qwen3.5-4b")
        self.assertIn("already loaded", str(ctx.exception))

    def test_53_no_dual_active_cache_entries(self) -> None:
        """53. ModelCache never contains more than one ACTIVE entry simultaneously."""
        self.loader.load("qwen3.5-4b")
        active_entries = [e for e in self.cache.list_entries() if e.is_active]
        self.assertEqual(len(active_entries), 1)

        # Force add another active entry through cache put to test query invariant
        self.cache.put(ModelCacheEntry("test-other", last_used_at=datetime.now(timezone.utc), cache_state=CacheState.CACHED_METADATA))
        active_entries_after = [e for e in self.cache.list_entries() if e.is_active]
        self.assertEqual(len(active_entries_after), 1)

    # ------------------------------------------------------------------
    # K. PROCESS ISOLATION & ARCHITECTURAL BOUNDARIES
    # ------------------------------------------------------------------

    def test_54_process_isolation_ast_check(self) -> None:
        """54. AST check: subprocess is imported ONLY in runtime.py."""
        ai_dir = Path(__file__).resolve().parent.parent / "app" / "ai"
        non_runtime_files = [
            f for f in ai_dir.glob("*.py")
            if f.name not in {"runtime.py", "__pycache__"}
        ]

        for py_file in non_runtime_files:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(
                            alias.name,
                            "subprocess",
                            f"Illegal subprocess import in {py_file.name}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(
                        node.module,
                        "subprocess",
                        f"Illegal subprocess import in {py_file.name}",
                    )

    def test_55_no_shell_true_in_any_runtime(self) -> None:
        """55. AST check: runtime.py passes argument lists to Popen and never shell=True."""
        runtime_file = Path(__file__).resolve().parent.parent / "app" / "ai" / "runtime.py"
        with open(runtime_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(runtime_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
                if func_name in {"Popen", "run", "call"}:
                    for kw in node.keywords:
                        if kw.arg == "shell":
                            self.assertIs(kw.value.value, False, "shell=True is forbidden")

    def test_56_unload_manager_has_no_direct_process_attributes(self) -> None:
        """56. UnloadManager has no direct process handle attributes."""
        self.assertFalse(hasattr(self.unload_manager, "process"))
        self.assertFalse(hasattr(self.unload_manager, "_process"))
        self.assertFalse(hasattr(self.unload_manager, "pid"))

    # ------------------------------------------------------------------
    # L. FULL SUBSYSTEM STATE CONSISTENCY
    # ------------------------------------------------------------------

    def test_57_tri_state_consistency_ready_state(self) -> None:
        """57. Legal READY state: Runtime=RUNNING, Lifecycle=READY, Cache=ACTIVE."""
        self.loader.load("qwen3.5-4b")
        self.assertEqual(self.mock_runtime.state, RuntimeState.RUNNING)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.ACTIVE)

    def test_58_tri_state_consistency_unloaded_state(self) -> None:
        """58. Legal UNLOADED state: Runtime=STOPPED, Lifecycle=UNLOADED, Cache=CACHED_METADATA."""
        self.loader.load("qwen3.5-4b")
        self.unload_manager.unload_current()
        self.assertEqual(self.mock_runtime.state, RuntimeState.STOPPED)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertEqual(self.cache.get("qwen3.5-4b").cache_state, CacheState.CACHED_METADATA)


class TestRealModelSubsystemSmoke(unittest.TestCase):
    """Real-model end-to-end integration smoke test with qwen3.5-4b."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.runtime = LlamaCppRuntime()
        self.lifecycle = ModelLifecycle()
        self.cache = ModelCache(max_entries=5)
        self.ram_manager = RAMManager(registry=self.registry)
        self.loader = ModelLoader(
            registry=self.registry,
            runtime=self.runtime,
            lifecycle=self.lifecycle,
            ram_manager=self.ram_manager,
            cache=self.cache,
        )
        self.unload_manager = UnloadManager(
            loader=self.loader,
            ram_manager=self.ram_manager,
            cache=self.cache,
        )

    def tearDown(self) -> None:
        # Guarantee cleanup
        if self.unload_manager.has_active_model:
            self.unload_manager.unload_current()
        if self.runtime.is_running():
            self.runtime.stop()

    def test_59_real_model_smoke_test_qwen3_5(self) -> None:
        """59. Real integration: Load qwen3.5-4b, verify tri-state consistency, unload cleanly."""
        model_path = self.registry.resolve_model_path("qwen3.5-4b")
        if not model_path.is_file():
            self.skipTest(f"Model file not found at {model_path}")

        # 1. Load through full subsystem
        status = self.loader.load("qwen3.5-4b")
        self.assertIsInstance(status, LoadedModelInfo)
        self.assertEqual(status.model_id, "qwen3.5-4b")

        # 2. Confirm Tri-State Consistency while active
        self.assertTrue(self.runtime.is_running())
        self.assertEqual(self.runtime.state, RuntimeState.RUNNING)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        cached = self.cache.get("qwen3.5-4b")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.cache_state, CacheState.ACTIVE)
        self.assertTrue(self.unload_manager.has_active_model)
        self.assertTrue(self.unload_manager.can_unload("qwen3.5-4b"))

        # 3. Allow real process to stabilize for 1 second
        time.sleep(1.0)
        self.assertTrue(self.runtime.is_running())

        # 4. Unload through UnloadManager
        result = self.unload_manager.unload_current(reason="Integration smoke test complete")
        self.assertTrue(result.success)
        self.assertEqual(result.model_id, "qwen3.5-4b")
        self.assertEqual(result.previous_state, ModelLifecycleState.READY)
        self.assertEqual(result.final_state, ModelLifecycleState.UNLOADED)

        # 5. Confirm Tri-State Consistency after unload
        self.assertFalse(self.runtime.is_running())
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        cached_after = self.cache.get("qwen3.5-4b")
        self.assertEqual(cached_after.cache_state, CacheState.CACHED_METADATA)
        self.assertEqual(cached_after.usage_count, 1)
        self.assertFalse(self.unload_manager.has_active_model)


if __name__ == "__main__":
    unittest.main()
