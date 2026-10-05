"""Tests for WHIS Stage 3.2: Runtime Adapter.

All tests utilize mocked subprocess behavior to verify argument construction,
device offload policy, process lifecycle, and exception handling without
spawning real llama.cpp processes or loading model weights.
"""

from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from app.ai.model_manager import ModelDefinition, ModelRegistry, PROJECT_ROOT
from app.ai.runtime import (
    DevicePolicy,
    GenerationConfig,
    InferenceResult,
    LlamaCppRuntime,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeConfig,
    RuntimeInferenceError,
    RuntimeProcessError,
    RuntimeStartError,
    RuntimeState,
)


class TestRuntimeAdapter(unittest.TestCase):
    """Test suite for LlamaCppRuntime and RuntimeAdapter abstraction."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.text_model = self.registry.get("qwen3.5-4b")
        self.vl_model = self.registry.get("qwen3-vl-4b")
        self.ocr_model = self.registry.get("glm-ocr")

        # Create a fresh runtime instance for each test
        self.runtime = LlamaCppRuntime()

    def tearDown(self) -> None:
        # Guarantee no real or mock process remains running
        if self.runtime.is_running():
            self.runtime.stop()

    def test_01_command_construction_text_model(self) -> None:
        """1. RuntimeAdapter can construct a command for a normal text model."""
        cmd = self.runtime.build_command(self.text_model)

        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "llama-cli")
        self.assertIn("-m", cmd)

        m_idx = cmd.index("-m")
        resolved_path = cmd[m_idx + 1]
        self.assertTrue(Path(resolved_path).is_absolute())
        self.assertTrue(resolved_path.endswith("Qwen3.5-4B-Q4_K_M.gguf"))

        # Text model must not include --mmproj
        self.assertNotIn("--mmproj", cmd)

        # Context length flag
        self.assertIn("-c", cmd)
        c_idx = cmd.index("-c")
        self.assertEqual(cmd[c_idx + 1], str(self.text_model.default_context))

    def test_02_command_construction_multimodal_model(self) -> None:
        """2. RuntimeAdapter includes --mmproj for a multimodal model."""
        cmd = self.runtime.build_command(self.vl_model)

        self.assertIsInstance(cmd, list)
        self.assertIn("-m", cmd)
        self.assertIn("--mmproj", cmd)

        mm_idx = cmd.index("--mmproj")
        resolved_mmproj = cmd[mm_idx + 1]
        self.assertTrue(Path(resolved_mmproj).is_absolute())
        self.assertTrue(resolved_mmproj.endswith("mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"))

        # Also verify OCR model with mmproj
        ocr_cmd = self.runtime.build_command(self.ocr_model)
        self.assertIn("--mmproj", ocr_cmd)
        ocr_mm_idx = ocr_cmd.index("--mmproj")
        self.assertTrue(ocr_cmd[ocr_mm_idx + 1].endswith("mmproj-GLM-OCR-Q8_0.gguf"))

    def test_03_cpu_safe_configuration(self) -> None:
        """3. CPU-safe configuration is represented correctly."""
        # Default config must be CPU_ONLY with 0 GPU layers offloaded
        self.assertEqual(self.runtime.config.device_policy, DevicePolicy.CPU_ONLY)
        self.assertEqual(self.runtime.resolve_ngl(), 0)

        cmd = self.runtime.build_command(self.text_model)
        self.assertIn("-ngl", cmd)
        ngl_idx = cmd.index("-ngl")
        self.assertEqual(cmd[ngl_idx + 1], "0")

        # AUTO policy on development hardware (Intel UHD) must also resolve to CPU-safe 0 ngl
        auto_runtime = LlamaCppRuntime(config=RuntimeConfig(device_policy=DevicePolicy.AUTO))
        self.assertEqual(auto_runtime.resolve_ngl(), 0)
        auto_cmd = auto_runtime.build_command(self.text_model)
        self.assertEqual(auto_cmd[auto_cmd.index("-ngl") + 1], "0")

        # Configurable GPU offload policy
        gpu_runtime = LlamaCppRuntime(config=RuntimeConfig(device_policy=DevicePolicy.GPU_OFFLOAD, ngl=33))
        self.assertEqual(gpu_runtime.resolve_ngl(), 33)
        gpu_cmd = gpu_runtime.build_command(self.text_model)
        self.assertEqual(gpu_cmd[gpu_cmd.index("-ngl") + 1], "33")

    @patch("subprocess.Popen")
    def test_04_process_starts_using_argument_list(self, mock_popen: MagicMock) -> None:
        """4. Process starts using argument-list subprocess invocation."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        self.runtime.start(self.text_model)

        self.assertTrue(mock_popen.called)
        args, kwargs = mock_popen.call_args

        # Command must be a list of strings, never a raw shell string
        self.assertIsInstance(args[0], list)
        self.assertEqual(args[0][0], "llama-cli")
        self.assertIn("-m", args[0])

        # shell=True must not be used
        self.assertFalse(kwargs.get("shell", False))

        # Pipes must be configured for stdout and stderr capture
        self.assertEqual(kwargs.get("stdout"), subprocess.PIPE)
        self.assertEqual(kwargs.get("stderr"), subprocess.PIPE)

    @patch("subprocess.Popen")
    def test_05_runtime_state_transitions(self, mock_popen: MagicMock) -> None:
        """5. Runtime state changes correctly: STOPPED -> STARTING -> RUNNING -> STOPPED."""
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        self.runtime.start(self.text_model)

        self.assertEqual(self.runtime.state, RuntimeState.RUNNING)
        self.assertEqual(self.runtime.current_model.id, "qwen3.5-4b")

        self.runtime.stop()
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)
        self.assertIsNone(self.runtime.current_model)

    @patch("subprocess.Popen")
    def test_06_is_running_reports_correctly(self, mock_popen: MagicMock) -> None:
        """6. is_running() correctly reports process state."""
        self.assertFalse(self.runtime.is_running())

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        self.runtime.start(self.text_model)
        self.assertTrue(self.runtime.is_running())

        # Simulate process exiting in background
        mock_proc.poll.return_value = 0
        self.assertFalse(self.runtime.is_running())
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)

    @patch("subprocess.Popen")
    def test_07_stop_terminates_process(self, mock_popen: MagicMock) -> None:
        """7. stop() terminates the process."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        self.runtime.start(self.text_model)
        self.assertTrue(self.runtime.is_running())

        self.runtime.stop()

        self.assertTrue(mock_proc.terminate.called)
        self.assertFalse(self.runtime.is_running())
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)
        self.assertIsNone(self.runtime.process)

    @patch("subprocess.Popen")
    def test_08_failed_process_startup_produces_runtime_exception(self, mock_popen: MagicMock) -> None:
        """8. Failed process startup produces a clear runtime exception."""
        # Case A: Executable not found on PATH
        mock_popen.side_effect = FileNotFoundError("llama-cli executable not found")

        with self.assertRaises(RuntimeStartError) as ctx:
            self.runtime.start(self.text_model)

        self.assertIn("not found", str(ctx.exception).lower())
        self.assertEqual(self.runtime.state, RuntimeState.FAILED)
        self.assertIsNone(self.runtime.current_model)

        # Case B: Immediate exit with non-zero code
        mock_popen.side_effect = None
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stderr.read.return_value = "unknown option --invalid"
        mock_popen.return_value = mock_proc

        with self.assertRaises(RuntimeStartError) as ctx2:
            self.runtime.start(self.text_model)

        self.assertEqual(ctx2.exception.returncode, 1)
        self.assertEqual(self.runtime.state, RuntimeState.FAILED)

    @patch("subprocess.Popen")
    def test_09_stderr_captured_on_startup_failure(self, mock_popen: MagicMock) -> None:
        """9. stderr is captured when startup fails."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 2
        mock_proc.stderr.read.return_value = "error: failed to load model: magic number mismatch"
        mock_popen.return_value = mock_proc

        with self.assertRaises(RuntimeStartError) as ctx:
            self.runtime.start(self.text_model)

        self.assertIn("magic number mismatch", ctx.exception.stderr)
        self.assertEqual(self.runtime.last_stderr, "error: failed to load model: magic number mismatch")
        self.assertEqual(ctx.exception.model_id, "qwen3.5-4b")
        self.assertEqual(ctx.exception.returncode, 2)

    @patch("subprocess.Popen")
    def test_10_starting_already_running_runtime_handled_safely(self, mock_popen: MagicMock) -> None:
        """10. Starting an already-running runtime is handled safely."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        self.runtime.start(self.text_model)
        self.assertTrue(self.runtime.is_running())

        # Attempting to start another model while already running raises RuntimeProcessError
        with self.assertRaises(RuntimeProcessError) as ctx:
            self.runtime.start(self.vl_model, restart_if_running=False)

        self.assertIn("already running", str(ctx.exception).lower())
        # First process remains active and untouched
        self.assertEqual(self.runtime.current_model.id, "qwen3.5-4b")

        # With restart_if_running=True, the active model is cleanly stopped and restarted
        self.runtime.start(self.vl_model, restart_if_running=True)
        self.assertEqual(self.runtime.current_model.id, "qwen3-vl-4b")

    def test_11_paths_containing_spaces_handled_correctly(self) -> None:
        """11. Paths containing spaces are handled correctly as individual arguments."""
        spaced_model = ModelDefinition(
            id="spaced-model",
            display_name="Spaced Model",
            type="llm",
            model_path="models/my custom directory/model with spaces.gguf",
            mmproj_path="models/my custom directory/projector with spaces.gguf",
            capabilities=["chat"],
            role="test",
            status="verified",
            runtime="llama.cpp",
            default_context=2048,
        )

        cmd = self.runtime.build_command(spaced_model)

        # Path must be a discrete element in the command array containing spaces
        m_idx = cmd.index("-m")
        model_arg = cmd[m_idx + 1]
        self.assertIn("model with spaces.gguf", model_arg)
        self.assertTrue(Path(model_arg).is_absolute())

        mm_idx = cmd.index("--mmproj")
        mmproj_arg = cmd[mm_idx + 1]
        self.assertIn("projector with spaces.gguf", mmproj_arg)
        self.assertTrue(Path(mmproj_arg).is_absolute())

    def test_12_no_orphan_processes_left(self) -> None:
        """12. Tests do not leave real llama.cpp processes running."""
        # Ensure runtime state clean
        self.assertFalse(self.runtime.is_running())
        self.assertEqual(self.runtime.state, RuntimeState.STOPPED)

    def test_custom_executable_configuration(self) -> None:
        """Runtime allows configurable executable name or path."""
        custom_runtime = LlamaCppRuntime(config=RuntimeConfig(executable="llama-server"))
        cmd = custom_runtime.build_command(self.text_model)
        self.assertEqual(cmd[0], "llama-server")

    def test_threads_configuration(self) -> None:
        """Runtime passes thread count flag when configured."""
        custom_runtime = LlamaCppRuntime(config=RuntimeConfig(threads=4))
        cmd = custom_runtime.build_command(self.text_model)
        self.assertIn("-t", cmd)
        t_idx = cmd.index("-t")
        self.assertEqual(cmd[t_idx + 1], "4")

    def test_generate_command_construction(self) -> None:
        """Runtime constructs correct arguments for one-shot inference."""
        cfg = GenerationConfig(max_new_tokens=64, temperature=0.7, stop_sequences=["User:"])
        cmd = self.runtime._build_generate_command(self.text_model, "Test prompt", cfg)

        self.assertIn("-m", cmd)
        self.assertIn("-p", cmd)
        p_idx = cmd.index("-p")
        self.assertEqual(cmd[p_idx + 1], "Test prompt")
        self.assertIn("-n", cmd)
        n_idx = cmd.index("-n")
        self.assertEqual(cmd[n_idx + 1], "64")
        self.assertIn("--temp", cmd)
        temp_idx = cmd.index("--temp")
        self.assertEqual(cmd[temp_idx + 1], "0.7")
        self.assertIn("--single-turn", cmd)
        self.assertIn("--no-display-prompt", cmd)
        self.assertIn("-r", cmd)
        r_idx = cmd.index("-r")
        self.assertEqual(cmd[r_idx + 1], "User:")

    def test_clean_output(self) -> None:
        """_clean_output strips ASCII banner, prompt echo, and timing stats."""
        raw = (
            "\n\nLoading model... \n\n▄▄ ▄▄\n██ ██\n\n"
            "> Hello world\n\n"
            "This is the actual model answer.\n\n"
            "[ Prompt: 7.4 t/s | Generation: 5.1 t/s ]\n\n"
            "Exiting...\n"
        )
        cleaned = LlamaCppRuntime._clean_output(raw)
        self.assertEqual(cleaned, "This is the actual model answer.")

    @patch("subprocess.run")
    def test_generate_success_mocked(self, mock_run: MagicMock) -> None:
        """Test successful one-shot inference execution with mocked subprocess."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n> What is 1+1?\n2\n[ Prompt: 5 t/s ]\nExiting...",
            stderr="",
        )

        result = self.runtime.generate(self.text_model, "What is 1+1?")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "2")
        self.assertEqual(result.model_id, self.text_model.id)
        self.assertGreater(result.prompt_tokens_estimate, 0)
        self.assertGreaterEqual(result.generated_tokens_estimate, 0)

    @patch("subprocess.run")
    def test_generate_timeout_mocked(self, mock_run: MagicMock) -> None:
        """Test timeout during one-shot inference raises RuntimeInferenceError."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["llama-cli"], timeout=5.0)

        with self.assertRaises(RuntimeInferenceError) as ctx:
            self.runtime.generate(self.text_model, "Test", config=GenerationConfig(timeout=5.0))
        self.assertIn("timed out", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

