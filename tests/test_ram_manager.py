"""Tests for WHIS Stage 3.5: RAM Manager.

Comprehensive unit tests verifying physical memory measurement, immutable snapshots,
model memory estimation (weights, projector, KV cache, overhead), configurable safety
reserves, safety evaluations (SAFE, WARNING, UNSAFE), and ModelLoader integration.
"""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock

from app.ai.loader import ModelLoader
from app.ai.model_manager import ModelRegistry
from app.ai.ram_manager import (
    DEFAULT_SAFETY_RESERVE_BYTES,
    InsufficientMemoryError,
    MemoryDecision,
    MemoryProvider,
    MemorySnapshot,
    ModelMemoryEstimate,
    RAMManager,
    SafetyEvaluation,
    SystemMemoryProvider,
)
from app.ai.runtime import LlamaCppRuntime, RuntimeAdapter, RuntimeConfig, RuntimeState


class MockMemoryProvider(MemoryProvider):
    """Custom memory provider for deterministic test scenarios."""

    def __init__(self, total_bytes: int, available_bytes: int) -> None:
        self.total_bytes = total_bytes
        self.available_bytes = available_bytes

    def get_memory_snapshot(self) -> MemorySnapshot:
        used = self.total_bytes - self.available_bytes
        percent = (used / self.total_bytes) * 100.0 if self.total_bytes > 0 else 0.0
        return MemorySnapshot(
            total_bytes=self.total_bytes,
            available_bytes=self.available_bytes,
            used_bytes=used,
            percent_used=round(percent, 2),
            timestamp=datetime.now(timezone.utc),
        )

    def get_process_memory(self, pid: int) -> int:
        return 150 * 1024 * 1024  # 150 MB


class TestRAMManager(unittest.TestCase):
    """Test suite for RAMManager, MemoryProvider, and memory safety evaluation."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.ram_manager = RAMManager(registry=self.registry)

    def test_01_system_memory_provider_real_snapshot(self) -> None:
        """1. SystemMemoryProvider captures physical system memory with timezone-aware UTC timestamp."""
        provider = SystemMemoryProvider()
        snapshot = provider.get_memory_snapshot()

        self.assertIsInstance(snapshot, MemorySnapshot)
        self.assertGreater(snapshot.total_bytes, 0)
        self.assertGreater(snapshot.available_bytes, 0)
        self.assertEqual(snapshot.used_bytes, snapshot.total_bytes - snapshot.available_bytes)
        self.assertGreaterEqual(snapshot.percent_used, 0.0)
        self.assertLessEqual(snapshot.percent_used, 100.0)
        self.assertIsNotNone(snapshot.timestamp.tzinfo)
        self.assertEqual(snapshot.timestamp.tzinfo, timezone.utc)

    def test_02_memory_snapshot_consistency(self) -> None:
        """2. MemorySnapshot properties (GB and arithmetic) are consistent."""
        total = 16 * 1024 * 1024 * 1024
        available = 6 * 1024 * 1024 * 1024
        used = total - available

        snap = MemorySnapshot(
            total_bytes=total,
            available_bytes=available,
            used_bytes=used,
            percent_used=62.5,
            timestamp=datetime.now(timezone.utc),
        )

        self.assertEqual(snap.total_gb, 16.0)
        self.assertEqual(snap.available_gb, 6.0)
        self.assertEqual(snap.used_gb, 10.0)

    def test_03_model_memory_estimation_not_equal_to_file_size(self) -> None:
        """3. ModelMemoryEstimate includes weights, KV cache, and runtime overhead (not just file size)."""
        estimate = self.ram_manager.estimate_model_memory("qwen3.5-4b")

        self.assertIsInstance(estimate, ModelMemoryEstimate)
        self.assertEqual(estimate.model_id, "qwen3.5-4b")
        self.assertGreater(estimate.weight_bytes, 0)
        self.assertGreater(estimate.estimated_kv_cache_bytes, 0)
        self.assertGreater(estimate.estimated_runtime_overhead_bytes, 0)

        # Critical requirement: estimated total must NOT simply equal weight file size
        self.assertGreater(estimate.estimated_total_bytes, estimate.weight_bytes)
        self.assertEqual(
            estimate.estimated_total_bytes,
            estimate.weight_bytes
            + estimate.projector_bytes
            + estimate.estimated_kv_cache_bytes
            + estimate.estimated_runtime_overhead_bytes,
        )
        self.assertEqual(estimate.confidence, "MEDIUM")

    def test_04_context_tokens_affects_estimation(self) -> None:
        """4. Custom context_tokens parameter scales the estimated KV cache size."""
        est_2k = self.ram_manager.estimate_model_memory("qwen3.5-4b", context_tokens=2048)
        est_8k = self.ram_manager.estimate_model_memory("qwen3.5-4b", context_tokens=8192)

        self.assertGreater(est_8k.estimated_kv_cache_bytes, est_2k.estimated_kv_cache_bytes)
        self.assertGreater(est_8k.estimated_total_bytes, est_2k.estimated_total_bytes)

    def test_05_multimodal_model_includes_projector(self) -> None:
        """5. Multimodal model memory estimation incorporates projector file size."""
        vl_estimate = self.ram_manager.estimate_model_memory("qwen3-vl-4b")

        self.assertGreater(vl_estimate.projector_bytes, 0)
        self.assertIn("projector", vl_estimate.notes.lower())
        self.assertGreater(
            vl_estimate.estimated_total_bytes,
            vl_estimate.weight_bytes + vl_estimate.projector_bytes,
        )

    def test_06_safety_evaluation_safe_decision(self) -> None:
        """6. Model safely fits within available RAM minus safety reserve -> SAFE."""
        # 16 GB total, 10 GB available, 2 GB reserve -> 8 GB safe available.
        # qwen3.5-4b requires ~3.5 GB -> SAFE.
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=10 * 1024 * 1024 * 1024,
        )
        mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

        evaluation = mgr.evaluate_safety("qwen3.5-4b")
        self.assertEqual(evaluation.decision, MemoryDecision.SAFE)
        self.assertTrue(evaluation.is_safe)
        self.assertTrue(evaluation.is_executable)
        self.assertTrue(mgr.can_load("qwen3.5-4b"))

    def test_07_safety_evaluation_warning_decision(self) -> None:
        """7. Model fits in available RAM but encroaches on safety reserve -> WARNING."""
        # 16 GB total, 4 GB available, 2 GB reserve -> 2 GB safe available.
        # qwen3.5-4b requires ~3.5 GB (fits in 4 GB, but encroaches into reserve) -> WARNING.
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=4 * 1024 * 1024 * 1024,
        )
        mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

        evaluation = mgr.evaluate_safety("qwen3.5-4b")
        self.assertEqual(evaluation.decision, MemoryDecision.WARNING)
        self.assertFalse(evaluation.is_safe)
        self.assertTrue(evaluation.is_executable)
        self.assertTrue(mgr.can_load("qwen3.5-4b", allow_warning=True))
        self.assertFalse(mgr.can_load("qwen3.5-4b", allow_warning=False))

    def test_08_safety_evaluation_unsafe_decision(self) -> None:
        """8. Model exceeds total available RAM -> UNSAFE."""
        # 16 GB total, 2 GB available -> qwen3.5-4b (~3.5 GB) cannot fit -> UNSAFE.
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=2 * 1024 * 1024 * 1024,
        )
        mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

        evaluation = mgr.evaluate_safety("qwen3.5-4b")
        self.assertEqual(evaluation.decision, MemoryDecision.UNSAFE)
        self.assertFalse(evaluation.is_safe)
        self.assertFalse(evaluation.is_executable)
        self.assertFalse(mgr.can_load("qwen3.5-4b"))

    def test_09_qwen3_coder_30b_evaluated_honestly_as_unsafe_on_16gb_baseline(self) -> None:
        """9. Qwen3-Coder 30B is evaluated honestly as UNSAFE on standard 16 GB RAM baseline."""
        # Standard development machine state: 16 GB total, 8 GB available
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=8 * 1024 * 1024 * 1024,
        )
        mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

        evaluation = mgr.evaluate_safety("qwen3-coder-30b")
        self.assertEqual(evaluation.decision, MemoryDecision.UNSAFE)
        self.assertIn("exceeds", evaluation.reason.lower())
        self.assertFalse(mgr.can_load("qwen3-coder-30b"))

    def test_10_configurable_safety_reserve(self) -> None:
        """10. Configurable safety reserve adjusts safe capacity."""
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=5 * 1024 * 1024 * 1024,
        )
        # With 1 GB reserve: 5 GB available - 1 GB reserve = 4 GB safe available -> qwen3.5-4b (~3.5 GB) is SAFE
        mgr_small_reserve = RAMManager(
            registry=self.registry,
            memory_provider=mock_provider,
            safety_reserve_bytes=1 * 1024 * 1024 * 1024,
        )
        eval_small = mgr_small_reserve.evaluate_safety("qwen3.5-4b")
        self.assertEqual(eval_small.decision, MemoryDecision.SAFE)

        # With 3 GB reserve: 5 GB available - 3 GB reserve = 2 GB safe available -> qwen3.5-4b (~3.5 GB) is WARNING
        mgr_large_reserve = RAMManager(
            registry=self.registry,
            memory_provider=mock_provider,
            safety_reserve_bytes=3 * 1024 * 1024 * 1024,
        )
        eval_large = mgr_large_reserve.evaluate_safety("qwen3.5-4b")
        self.assertEqual(eval_large.decision, MemoryDecision.WARNING)

    def test_11_model_loader_rejects_unsafe_model_load(self) -> None:
        """11. ModelLoader integrated with RAMManager rejects UNSAFE model loads."""
        # Mock provider with very low RAM: 1.5 GB available
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=int(1.5 * 1024 * 1024 * 1024),
        )
        ram_mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        loader = ModelLoader(
            registry=self.registry,
            runtime=mock_runtime,
            ram_manager=ram_mgr,
        )

        with self.assertRaises(InsufficientMemoryError) as ctx:
            loader.load("qwen3.5-4b", check_ram=True)

        self.assertEqual(ctx.exception.model_id, "qwen3.5-4b")
        self.assertEqual(ctx.exception.evaluation.decision, MemoryDecision.UNSAFE)
        # Runtime must not have been started
        self.assertFalse(mock_runtime.start.called)
        self.assertFalse(loader.is_loaded())

    def test_12_model_loader_accepts_safe_model_load(self) -> None:
        """12. ModelLoader integrated with RAMManager permits SAFE model loads."""
        mock_provider = MockMemoryProvider(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=10 * 1024 * 1024 * 1024,
        )
        ram_mgr = RAMManager(registry=self.registry, memory_provider=mock_provider)

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
            ram_manager=ram_mgr,
        )

        status = loader.load("qwen3.5-4b", check_ram=True)
        self.assertTrue(loader.is_loaded())
        self.assertEqual(status.model_id, "qwen3.5-4b")


if __name__ == "__main__":
    unittest.main()
