"""Tests for WHIS Stage 3.4: Model State & Lifecycle.

Comprehensive unit tests verifying the ModelLifecycle state machine, legal and
illegal transitions, timezone-aware UTC timestamps, RuntimeState separation,
runtime synchronization, and integration with ModelLoader.
"""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock

from app.ai.lifecycle import (
    InvalidStateTransitionError,
    ModelLifecycle,
    ModelLifecycleError,
    ModelLifecycleState,
    ModelRuntimeStatus,
)
from app.ai.loader import ModelLoadError, ModelLoader
from app.ai.model_manager import ModelDefinition, ModelRegistry
from app.ai.runtime import (
    DevicePolicy,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeConfig,
    RuntimeStartError,
    RuntimeState,
)


class TestModelLifecycle(unittest.TestCase):
    """Test suite for ModelLifecycle state machine and its integration."""

    def setUp(self) -> None:
        self.lifecycle = ModelLifecycle()

    def test_01_initial_state_is_unloaded(self) -> None:
        """1. Initial state is UNLOADED."""
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertIsNone(self.lifecycle.model_id)
        self.assertFalse(self.lifecycle.is_ready())
        self.assertFalse(self.lifecycle.is_busy())
        self.assertFalse(self.lifecycle.is_loaded())

    def test_02_begin_loading_transitions_unloaded_to_loading(self) -> None:
        """2. begin_loading() transitions: UNLOADED -> LOADING."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.LOADING)
        self.assertEqual(self.lifecycle.model_id, "qwen3.5-4b")
        self.assertFalse(self.lifecycle.is_ready())
        self.assertFalse(self.lifecycle.is_busy())
        self.assertFalse(self.lifecycle.is_loaded())

    def test_03_mark_ready_transitions_loading_to_ready(self) -> None:
        """3. mark_ready() transitions: LOADING -> READY."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertTrue(self.lifecycle.is_ready())
        self.assertTrue(self.lifecycle.is_loaded())
        self.assertFalse(self.lifecycle.is_busy())

    def test_04_begin_use_transitions_ready_to_busy(self) -> None:
        """4. begin_use() transitions: READY -> BUSY."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.begin_use()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.BUSY)
        self.assertTrue(self.lifecycle.is_busy())
        self.assertTrue(self.lifecycle.is_loaded())
        self.assertFalse(self.lifecycle.is_ready())

    def test_05_end_use_transitions_busy_to_ready(self) -> None:
        """5. end_use() transitions: BUSY -> READY."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.begin_use()
        self.lifecycle.end_use()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.READY)
        self.assertTrue(self.lifecycle.is_ready())
        self.assertFalse(self.lifecycle.is_busy())

    def test_06_begin_stopping_transitions_ready_to_stopping(self) -> None:
        """6. begin_stopping() transitions: READY -> STOPPING."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.begin_stopping()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.STOPPING)
        self.assertFalse(self.lifecycle.is_ready())
        self.assertFalse(self.lifecycle.is_loaded())

    def test_07_mark_unloaded_transitions_stopping_to_unloaded(self) -> None:
        """7. mark_unloaded() transitions: STOPPING -> UNLOADED."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.begin_stopping()
        self.lifecycle.mark_unloaded()
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertIsNone(self.lifecycle.model_id)

    def test_08_loading_failure_transitions_to_failed(self) -> None:
        """8. Loading failure: LOADING -> FAILED."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_failed("Executable not found on PATH")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertEqual(self.lifecycle.failure_reason, "Executable not found on PATH")
        self.assertFalse(self.lifecycle.is_loaded())

    def test_09_runtime_failure_transitions_to_failed(self) -> None:
        """9. Runtime failure: READY -> FAILED."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.mark_failed("Subprocess crashed with SIGSEGV")
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertEqual(self.lifecycle.failure_reason, "Subprocess crashed with SIGSEGV")
        self.assertFalse(self.lifecycle.is_ready())

    def test_10_invalid_transitions_are_rejected(self) -> None:
        """10. Invalid transitions raise InvalidStateTransitionError with context."""
        # UNLOADED -> BUSY is invalid
        with self.assertRaises(InvalidStateTransitionError) as ctx:
            self.lifecycle.begin_use()
        self.assertIn("UNLOADED", str(ctx.exception))
        self.assertIn("BUSY", str(ctx.exception))

        # UNLOADED -> READY is invalid
        with self.assertRaises(InvalidStateTransitionError):
            self.lifecycle.mark_ready()

        # READY -> LOADING is invalid
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        with self.assertRaises(InvalidStateTransitionError) as ctx2:
            self.lifecycle.begin_loading("qwen3.5-4b")
        self.assertIn("READY", str(ctx2.exception))
        self.assertIn("LOADING", str(ctx2.exception))

        # LOADING -> BUSY is invalid
        fresh_lc = ModelLifecycle()
        fresh_lc.begin_loading("qwen3.5-4b")
        with self.assertRaises(InvalidStateTransitionError):
            fresh_lc.begin_use()

    def test_11_last_used_at_updates_on_begin_use(self) -> None:
        """11. last_used_at updates on begin_use()."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.assertIsNone(self.lifecycle.last_used_at)

        self.lifecycle.begin_use()
        t1 = self.lifecycle.last_used_at
        self.assertIsNotNone(t1)
        self.assertEqual(t1.tzinfo, timezone.utc)
        self.assertEqual(self.lifecycle.operation_count, 1)

        self.lifecycle.end_use()
        self.lifecycle.begin_use()
        t2 = self.lifecycle.last_used_at
        self.assertGreaterEqual(t2, t1)
        self.assertEqual(self.lifecycle.operation_count, 2)

    def test_12_loaded_at_is_recorded(self) -> None:
        """12. loaded_at is recorded as timezone-aware UTC datetime."""
        self.assertIsNone(self.lifecycle.loaded_at)
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.assertIsNotNone(self.lifecycle.loaded_at)
        self.assertEqual(self.lifecycle.loaded_at.tzinfo, timezone.utc)

    def test_13_ready_at_is_recorded(self) -> None:
        """13. ready_at is recorded as timezone-aware UTC datetime."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.assertIsNone(self.lifecycle.ready_at)
        self.lifecycle.mark_ready()
        self.assertIsNotNone(self.lifecycle.ready_at)
        self.assertEqual(self.lifecycle.ready_at.tzinfo, timezone.utc)
        self.assertGreaterEqual(self.lifecycle.ready_at, self.lifecycle.loaded_at)

    def test_14_stopped_at_is_recorded(self) -> None:
        """14. stopped_at is recorded on mark_unloaded()."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()
        self.lifecycle.begin_stopping()
        self.assertIsNone(self.lifecycle.stopped_at)
        self.lifecycle.mark_unloaded()
        self.assertIsNotNone(self.lifecycle.stopped_at)
        self.assertEqual(self.lifecycle.stopped_at.tzinfo, timezone.utc)

    def test_15_failure_reason_is_recorded(self) -> None:
        """15. failure_reason and error_count are recorded on failure."""
        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_failed("CUDA out of memory")
        self.assertEqual(self.lifecycle.failure_reason, "CUDA out of memory")
        self.assertEqual(self.lifecycle.error_count, 1)

    def test_16_runtime_state_and_lifecycle_state_remain_separate(self) -> None:
        """16. RuntimeState and ModelLifecycleState remain separate and non-overlapping."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.state = RuntimeState.RUNNING

        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()

        status = self.lifecycle.get_status(runtime=mock_runtime)
        self.assertEqual(status.lifecycle_state, ModelLifecycleState.READY)
        self.assertEqual(status.runtime_state, RuntimeState.RUNNING)
        # Ensure they are distinct enum types
        self.assertIsInstance(status.lifecycle_state, ModelLifecycleState)
        self.assertIsInstance(status.runtime_state, RuntimeState)
        self.assertNotEqual(status.lifecycle_state, status.runtime_state)

    def test_17_sync_with_runtime_handles_runtime_failure(self) -> None:
        """17. sync_with_runtime() detects runtime crash/failure and marks FAILED."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = True
        mock_runtime.state = RuntimeState.RUNNING

        self.lifecycle.begin_loading("qwen3.5-4b")
        self.lifecycle.mark_ready()

        # Simulate background process crash
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.FAILED
        mock_runtime.last_stderr = "Segmentation fault"

        self.lifecycle.sync_with_runtime(mock_runtime)
        self.assertEqual(self.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertEqual(self.lifecycle.failure_reason, "Segmentation fault")

    def test_18_model_loader_integration_updates_lifecycle_state(self) -> None:
        """18. ModelLoader integration correctly updates lifecycle state."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli", device_policy=DevicePolicy.CPU_ONLY)

        def fake_start(model: ModelDefinition) -> None:
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        mock_runtime.start.side_effect = fake_start

        loader = ModelLoader(runtime=mock_runtime)
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.UNLOADED)

        loader.load("qwen3.5-4b")
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.READY)
        self.assertEqual(loader.lifecycle.model_id, "qwen3.5-4b")

    def test_19_successful_load_results_in_ready_state(self) -> None:
        """19. Successful load results in: ModelState = READY."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        def fake_start(model: ModelDefinition) -> None:
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        mock_runtime.start.side_effect = fake_start

        loader = ModelLoader(runtime=mock_runtime)
        status = loader.load("qwen3.5-4b")

        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.READY)
        self.assertEqual(status.lifecycle_state, ModelLifecycleState.READY)
        self.assertTrue(loader.lifecycle.is_ready())

    def test_20_successful_unload_results_in_unloaded_state(self) -> None:
        """20. Successful unload results in: ModelState = UNLOADED."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        def fake_start(model: ModelDefinition) -> None:
            mock_runtime.is_running.return_value = True
            mock_runtime.state = RuntimeState.RUNNING

        def fake_stop(timeout: float = 5.0) -> None:
            mock_runtime.is_running.return_value = False
            mock_runtime.state = RuntimeState.STOPPED

        mock_runtime.start.side_effect = fake_start
        mock_runtime.stop.side_effect = fake_stop

        loader = ModelLoader(runtime=mock_runtime)
        loader.load("qwen3.5-4b")
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.READY)

        loader.unload()
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.UNLOADED)
        self.assertFalse(loader.lifecycle.is_loaded())

    def test_21_failed_load_does_not_leave_lifecycle_stuck_in_loading(self) -> None:
        """21. Failed load does not leave the lifecycle stuck in LOADING."""
        mock_runtime = MagicMock(spec=RuntimeAdapter)
        mock_runtime.is_running.return_value = False
        mock_runtime.state = RuntimeState.STOPPED
        mock_runtime.config = RuntimeConfig(executable="llama-cli")

        mock_runtime.start.side_effect = RuntimeStartError(
            "Instant termination",
            model_id="qwen3.5-4b",
            returncode=1,
            stderr="Invalid arguments",
        )

        loader = ModelLoader(runtime=mock_runtime)

        with self.assertRaises(ModelLoadError):
            loader.load("qwen3.5-4b")

        # Must be FAILED, definitely NOT stuck in LOADING
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.FAILED)
        self.assertIsNotNone(loader.lifecycle.failure_reason)

        # Unloading recovers to UNLOADED
        loader.unload()
        self.assertEqual(loader.lifecycle.state, ModelLifecycleState.UNLOADED)

    def test_22_no_real_model_required_for_unit_tests(self) -> None:
        """22. Verifies lifecycle behavior completes deterministically without real model."""
        lc = ModelLifecycle()
        lc.begin_loading("virtual-model")
        lc.mark_ready()
        lc.begin_use()
        lc.end_use()
        lc.begin_stopping()
        lc.mark_unloaded()
        self.assertEqual(lc.state, ModelLifecycleState.UNLOADED)
        self.assertEqual(lc.operation_count, 1)


if __name__ == "__main__":
    unittest.main()
