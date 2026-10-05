"""Unit tests for WHIS InferenceEngine."""

import unittest
from unittest.mock import MagicMock, patch

from app.ai.context import ContextError, ContextManager, ContextOverflowError
from app.ai.inference import (
    DEFAULT_MODEL_ID,
    InferenceEngine,
    InferenceError,
    InferenceResponse,
)
from app.ai.lifecycle import ModelLifecycle, ModelLifecycleState
from app.ai.loader import ModelLoader
from app.ai.model_manager import ModelDefinition, ModelRegistry
from app.ai.ram_manager import RAMManager
from app.ai.runtime import (
    GenerationConfig,
    InferenceResult,
    RuntimeAdapter,
    RuntimeInferenceError,
    RuntimeState,
)
from app.ai.unload import UnloadManager


class FakeRuntimeAdapter(RuntimeAdapter):
    """Controllable test double for RuntimeAdapter."""

    def __init__(self) -> None:
        self._state: RuntimeState = RuntimeState.STOPPED
        self._current_model: ModelDefinition | None = None
        self.generate_calls: list[dict] = []
        self.start_calls: list[str] = []
        self.stop_calls: int = 0
        self.mock_output: str = "This is a fake inference response."
        self.raise_on_generate: Exception | None = None

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def current_model(self) -> ModelDefinition | None:
        return self._current_model

    def is_running(self) -> bool:
        return self._state == RuntimeState.RUNNING

    def build_command(self, model: ModelDefinition) -> list[str]:
        return ["fake-llama", "-m", model.model_path]

    def start(self, model: ModelDefinition, restart_if_running: bool = False) -> None:
        self.start_calls.append(model.id)
        self._current_model = model
        self._state = RuntimeState.RUNNING

    def stop(self, timeout: float | None = None) -> None:
        self.stop_calls += 1
        self._current_model = None
        self._state = RuntimeState.STOPPED

    def generate(
        self,
        model: ModelDefinition,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> InferenceResult:
        self.generate_calls.append({"model_id": model.id, "prompt": prompt, "config": config})
        if self.raise_on_generate:
            raise self.raise_on_generate
        return InferenceResult(
            text=self.mock_output,
            model_id=model.id,
            success=True,
            elapsed_seconds=0.15,
            prompt_tokens_estimate=10,
            generated_tokens_estimate=8,
        )

    def embed(
        self,
        model: ModelDefinition,
        text: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class TestInferenceEngine(unittest.TestCase):
    """Test suite for InferenceEngine functionality."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.runtime = FakeRuntimeAdapter()
        self.lifecycle = ModelLifecycle()
        self.ram_manager = MagicMock(spec=RAMManager)
        # RAM manager returns safe decision
        mock_decision = MagicMock()
        mock_decision.is_safe = True
        mock_decision.is_warning = False
        self.ram_manager.evaluate_safety.return_value = mock_decision

        self.loader = ModelLoader(
            registry=self.registry,
            runtime=self.runtime,
            lifecycle=self.lifecycle,
            ram_manager=self.ram_manager,
        )
        self.unload_manager = UnloadManager(
            loader=self.loader,
            ram_manager=self.ram_manager,
        )
        self.context_manager = ContextManager(generation_tokens=128)
        self.engine = InferenceEngine(
            registry=self.registry,
            runtime=self.runtime,
            loader=self.loader,
            unload_manager=self.unload_manager,
            ram_manager=self.ram_manager,
            context_manager=self.context_manager,
        )

    def test_default_model_selection(self) -> None:
        """Verify default model selection when no arguments given."""
        selected = self.engine.select_model_id()
        self.assertEqual(selected, DEFAULT_MODEL_ID)

    def test_explicit_model_selection(self) -> None:
        """Explicit model ID overrides capability mapping and default."""
        selected = self.engine.select_model_id(explicit_model_id="lfm2.5-thinking-1.2b")
        self.assertEqual(selected, "lfm2.5-thinking-1.2b")

    def test_capability_model_selection(self) -> None:
        """Verify capability mapping."""
        self.assertEqual(self.engine.select_model_id(capability="reasoning"), "lfm2.5-thinking-1.2b")
        self.assertEqual(self.engine.select_model_id(capability="code"), "qwen3-coder-30b")
        self.assertEqual(self.engine.select_model_id(capability="vision"), "qwen3-vl-4b")
        self.assertEqual(self.engine.select_model_id(capability="ocr"), "glm-ocr")
        self.assertEqual(self.engine.select_model_id(capability="unknown_capability"), DEFAULT_MODEL_ID)

    def test_ask_basic_flow(self) -> None:
        """Test ask() generates a valid response and loads default model."""
        response = self.engine.ask("What is a hash map?")

        self.assertIsInstance(response, InferenceResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.text, "This is a fake inference response.")
        self.assertEqual(response.model_id, DEFAULT_MODEL_ID)
        self.assertFalse(response.model_reused)
        self.assertEqual(len(self.runtime.generate_calls), 1)
        self.assertIn("What is a hash map?", self.runtime.generate_calls[0]["prompt"])
        self.assertTrue(self.engine.is_model_loaded)
        self.assertEqual(self.engine.active_model_id, DEFAULT_MODEL_ID)

    def test_model_reuse(self) -> None:
        """Second call with the same model reuses the loaded instance."""
        res1 = self.engine.ask("Question 1")
        self.assertFalse(res1.model_reused)
        start_count = len(self.runtime.start_calls)

        res2 = self.engine.ask("Question 2")
        self.assertTrue(res2.model_reused)
        # No extra start call occurred
        self.assertEqual(len(self.runtime.start_calls), start_count)

    def test_model_switching_unloads_previous(self) -> None:
        """Switching models cleanly unloads previous model first."""
        res1 = self.engine.ask("Math question", model_id="qwen3.5-4b")
        self.assertEqual(self.engine.active_model_id, "qwen3.5-4b")

        # Now ask reasoning model
        res2 = self.engine.ask("Logic puzzle", capability="reasoning")
        self.assertEqual(res2.model_id, "lfm2.5-thinking-1.2b")
        self.assertFalse(res2.model_reused)
        self.assertEqual(self.engine.active_model_id, "lfm2.5-thinking-1.2b")
        # Ensure previous model was stopped
        self.assertGreaterEqual(self.runtime.stop_calls, 1)

    def test_empty_prompt_raises_context_error(self) -> None:
        """Empty or whitespace prompt raises ContextError."""
        with self.assertRaises(ContextError):
            self.engine.ask("")
        with self.assertRaises(ContextError):
            self.engine.ask("   \t  ")

    def test_unregistered_model_raises_inference_error(self) -> None:
        """Requesting an unregistered model raises InferenceError."""
        with self.assertRaises(InferenceError) as ctx:
            self.engine.ask("Hello", model_id="nonexistent-model-xyz")
        self.assertIn("not registered", str(ctx.exception))

    def test_runtime_inference_error_propagated(self) -> None:
        """Runtime failures during generation raise InferenceError."""
        self.runtime.raise_on_generate = RuntimeInferenceError("llama-cli crashed", model_id="qwen3.5-4b")

        with self.assertRaises(InferenceError) as ctx:
            self.engine.ask("Will this fail?")
        self.assertIn("llama-cli crashed", str(ctx.exception))

    def test_explicit_unload(self) -> None:
        """Calling unload() stops the active model."""
        self.engine.ask("Load model")
        self.assertTrue(self.engine.is_model_loaded)

        self.engine.unload()
        self.assertFalse(self.engine.is_model_loaded)
        self.assertIsNone(self.engine.active_model_id)


if __name__ == "__main__":
    unittest.main()
