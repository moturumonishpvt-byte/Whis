"""Tests for WHIS Stage 3.3: Model Loader.

Includes comprehensive unit tests with mocked runtime and one isolated,
controlled real-model integration test for qwen3.5-4b.
"""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from app.ai.loader import (
    LoadedModelInfo,
    ModelAlreadyLoadedError,
    ModelDisabledError,
    ModelLoadError,
    ModelLoader,
    ModelLoaderError,
)
from app.ai.model_manager import (
    ModelDefinition,
    ModelNotFoundError,
    ModelRegistry,
    PROJECT_ROOT,
)
from app.ai.runtime import (
    DevicePolicy,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeConfig,
    RuntimeStartError,
    RuntimeState,
)


class TestModelLoaderUnit(unittest.TestCase):
    """Unit test suite for ModelLoader orchestration using mocked runtime."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.mock_runtime = MagicMock(spec=RuntimeAdapter)
        self.mock_runtime.is_running.return_value = False
        self.mock_runtime.state = RuntimeState.STOPPED
        self.mock_runtime.config = RuntimeConfig(executable="llama-cli", device_policy=DevicePolicy.CPU_ONLY)

        self.loader = ModelLoader(registry=self.registry, runtime=self.mock_runtime)

    def test_01_load_success_and_returns_status(self) -> None:
        """1. ModelLoader resolves model, calls runtime.start, and returns LoadedModelInfo."""
        # Setup mock behavior when started
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start

        status = self.loader.load("qwen3.5-4b")

        self.assertIsInstance(status, LoadedModelInfo)
        self.assertEqual(status.model_id, "qwen3.5-4b")
        self.assertEqual(status.display_name, "Qwen3.5 4B")
        self.assertEqual(status.role, "main_brain")
        self.assertEqual(status.runtime_state, RuntimeState.RUNNING)
        self.assertEqual(status.device_policy, DevicePolicy.CPU_ONLY)
        self.assertEqual(status.executable, "llama-cli")
        self.assertTrue(self.mock_runtime.start.called)

    def test_02_is_loaded_and_loaded_model_reporting(self) -> None:
        """2. is_loaded() and loaded_model() correctly report active model status."""
        self.assertFalse(self.loader.is_loaded())
        self.assertFalse(self.loader.is_loaded("qwen3.5-4b"))
        self.assertIsNone(self.loader.loaded_model())

        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start
        self.loader.load("qwen3.5-4b")

        self.assertTrue(self.loader.is_loaded())
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))
        self.assertFalse(self.loader.is_loaded("qwen3-coder-30b"))

        model = self.loader.loaded_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.id, "qwen3.5-4b")

    def test_03_unload_cleans_up_state(self) -> None:
        """3. unload() terminates runtime and clears internal tracking."""
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        def fake_stop(timeout: float = 5.0) -> None:
            self.mock_runtime.is_running.return_value = False
            self.mock_runtime.state = RuntimeState.STOPPED

        self.mock_runtime.start.side_effect = fake_start
        self.mock_runtime.stop.side_effect = fake_stop

        self.loader.load("qwen3.5-4b")
        self.assertTrue(self.loader.is_loaded())

        self.loader.unload()

        self.assertTrue(self.mock_runtime.stop.called)
        self.assertFalse(self.loader.is_loaded())
        self.assertIsNone(self.loader.loaded_model())
        self.assertIsNone(self.loader.get_status())

    def test_04_unknown_model_raises_model_not_found(self) -> None:
        """4. Attempting to load an unregistered model ID raises ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError):
            self.loader.load("unknown-nonexistent-model")

        self.assertFalse(self.loader.is_loaded())
        self.assertFalse(self.mock_runtime.start.called)

    def test_05_disabled_model_raises_model_disabled_error(self) -> None:
        """5. Attempting to load a disabled model raises ModelDisabledError."""
        disabled_model = ModelDefinition(
            id="disabled-model",
            display_name="Disabled Model",
            type="llm",
            model_path="models/llama.cpp/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf",
            capabilities=["chat"],
            role="test",
            status="verified",
            runtime="llama.cpp",
            enabled=False,
        )

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get.return_value = disabled_model
        loader = ModelLoader(registry=mock_registry, runtime=self.mock_runtime)

        with self.assertRaises(ModelDisabledError) as ctx:
            loader.load("disabled-model")

        self.assertEqual(ctx.exception.model_id, "disabled-model")
        self.assertIn("disabled", str(ctx.exception).lower())
        self.assertFalse(self.mock_runtime.start.called)

    def test_06_missing_weights_file_raises_model_load_error(self) -> None:
        """6. Attempting to load a model with missing GGUF weights raises ModelLoadError."""
        fake_model = ModelDefinition(
            id="missing-weights-model",
            display_name="Missing Weights",
            type="llm",
            model_path="models/llama.cpp/nonexistent/missing.gguf",
            capabilities=["chat"],
            role="test",
            status="verified",
            runtime="llama.cpp",
            enabled=True,
        )

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get.return_value = fake_model
        mock_registry.resolve_model_path.return_value = Path("C:/nonexistent/missing.gguf")

        loader = ModelLoader(registry=mock_registry, runtime=self.mock_runtime)

        with self.assertRaises(ModelLoadError) as ctx:
            loader.load("missing-weights-model")

        self.assertIn("weights file not found", str(ctx.exception).lower())
        self.assertFalse(self.mock_runtime.start.called)

    def test_07_missing_projector_file_raises_model_load_error(self) -> None:
        """7. Attempting to load a multimodal model with missing projector file raises ModelLoadError."""
        fake_vlm = ModelDefinition(
            id="missing-projector-vlm",
            display_name="Missing Projector VLM",
            type="vlm",
            model_path="models/llama.cpp/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf",
            mmproj_path="models/llama.cpp/nonexistent/missing_mmproj.gguf",
            capabilities=["vision"],
            role="vision_understanding",
            status="verified",
            runtime="llama.cpp",
            enabled=True,
        )

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get.return_value = fake_vlm
        mock_registry.resolve_model_path.return_value = PROJECT_ROOT / "models" / "llama.cpp" / "qwen3.5-4b" / "Qwen3.5-4B-Q4_K_M.gguf"
        mock_registry.resolve_mmproj_path.return_value = Path("C:/nonexistent/missing_mmproj.gguf")

        loader = ModelLoader(registry=mock_registry, runtime=self.mock_runtime)

        with self.assertRaises(ModelLoadError) as ctx:
            loader.load("missing-projector-vlm")

        self.assertIn("projector file not found", str(ctx.exception).lower())
        self.assertFalse(self.mock_runtime.start.called)

    def test_08_already_loaded_same_model_raises_error(self) -> None:
        """8. Loading the same model while active raises ModelAlreadyLoadedError."""
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start
        self.loader.load("qwen3.5-4b")

        with self.assertRaises(ModelAlreadyLoadedError) as ctx:
            self.loader.load("qwen3.5-4b")

        self.assertEqual(ctx.exception.model_id, "qwen3.5-4b")
        self.assertIn("already loaded", str(ctx.exception).lower())

    def test_09_already_loaded_different_model_raises_error(self) -> None:
        """9. Loading another model while one is active raises ModelAlreadyLoadedError."""
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start
        self.loader.load("qwen3.5-4b")

        # Second model loading must be blocked
        with self.assertRaises(ModelAlreadyLoadedError) as ctx:
            self.loader.load("qwen3-coder-30b")

        self.assertIn("already active", str(ctx.exception).lower())
        self.assertIn("qwen3.5-4b", str(ctx.exception))
        # Active model must remain qwen3.5-4b
        self.assertEqual(self.loader.loaded_model().id, "qwen3.5-4b")

    def test_10_startup_failure_cleanup_and_recovery(self) -> None:
        """10. If runtime fails to launch, loader cleans up state and allows subsequent load."""
        self.mock_runtime.start.side_effect = RuntimeStartError(
            "Executable crashed on launch",
            model_id="qwen3.5-4b",
            returncode=1,
            stderr="Out of memory",
        )

        with self.assertRaises(ModelLoadError) as ctx:
            self.loader.load("qwen3.5-4b")

        self.assertIn("failed to start runtime", str(ctx.exception).lower())
        self.assertFalse(self.loader.is_loaded())
        self.assertIsNone(self.loader.loaded_model())

        # Recovery: subsequent load attempt should be permitted
        def fake_start_ok(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start_ok
        status = self.loader.load("qwen3.5-4b")
        self.assertTrue(self.loader.is_loaded())
        self.assertEqual(status.model_id, "qwen3.5-4b")

    def test_11_process_exit_in_background_syncs_state(self) -> None:
        """11. If the underlying subprocess dies in background, is_loaded() detects it and resets."""
        def fake_start(model: ModelDefinition) -> None:
            self.mock_runtime.is_running.return_value = True
            self.mock_runtime.state = RuntimeState.RUNNING

        self.mock_runtime.start.side_effect = fake_start
        self.loader.load("qwen3.5-4b")
        self.assertTrue(self.loader.is_loaded())

        # Simulate background termination
        self.mock_runtime.is_running.return_value = False
        self.mock_runtime.state = RuntimeState.STOPPED

        self.assertFalse(self.loader.is_loaded())
        self.assertIsNone(self.loader.loaded_model())
        self.assertIsNone(self.loader.get_status())


class TestRealModelLoaderIntegration(unittest.TestCase):
    """Isolated, controlled real-model integration test.

    Specifically verifies that qwen3.5-4b can be resolved via ModelRegistry,
    launched through ModelLoader / LlamaCppRuntime, verified running, and stopped
    cleanly with no orphan processes.
    """

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.runtime = LlamaCppRuntime()
        self.loader = ModelLoader(registry=self.registry, runtime=self.runtime)

    def tearDown(self) -> None:
        # Guarantee no orphan llama.cpp process is left running
        if self.loader.is_loaded():
            self.loader.unload()
        if self.runtime.is_running():
            self.runtime.stop()

    def test_real_qwen3_5_4b_lifecycle(self) -> None:
        """Controlled real integration test: Load qwen3.5-4b, verify process alive, and cleanly unload."""
        # 1. Resolve and verify model in registry
        model = self.registry.get("qwen3.5-4b")
        self.assertEqual(model.id, "qwen3.5-4b")
        self.assertTrue(self.registry.resolve_model_path("qwen3.5-4b").is_file())

        # 2. Start model via ModelLoader
        status = self.loader.load("qwen3.5-4b")

        # 3. Confirm process is alive and loader reports loaded
        self.assertIsInstance(status, LoadedModelInfo)
        self.assertEqual(status.model_id, "qwen3.5-4b")
        self.assertTrue(self.loader.is_loaded())
        self.assertTrue(self.loader.is_loaded("qwen3.5-4b"))
        self.assertEqual(self.loader.loaded_model().id, "qwen3.5-4b")
        self.assertTrue(self.runtime.is_running())

        # 4. Give the real process a brief moment to run stably
        time.sleep(1.0)
        self.assertTrue(self.runtime.is_running())

        # 5. Stop / unload the model
        self.loader.unload()

        # 6. Confirm no process remains and loader state is clean
        self.assertFalse(self.loader.is_loaded())
        self.assertIsNone(self.loader.loaded_model())
        self.assertFalse(self.runtime.is_running())


if __name__ == "__main__":
    unittest.main()
