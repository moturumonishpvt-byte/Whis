"""Runtime adapter for llama.cpp execution.

Provides an isolated abstraction layer for managing llama.cpp subprocesses
without leaking process or command-line specifics across the WHIS codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Union
import urllib.request

from app.ai.model_manager import PROJECT_ROOT, ModelDefinition
from app.core.exceptions import WHISError

logger = logging.getLogger(__name__)



class RuntimeState(str, Enum):
    """Lifecycle states of the runtime adapter."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


class DevicePolicy(str, Enum):
    """Device offloading policy for the runtime."""

    CPU_ONLY = "cpu_only"
    GPU_OFFLOAD = "gpu_offload"
    AUTO = "auto"


class RuntimeAdapterError(WHISError):
    """Base exception for runtime adapter failures."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        executable: Optional[str] = None,
        command: Optional[List[str]] = None,
        returncode: Optional[int] = None,
        stderr: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.executable = executable
        self.command = command
        self.returncode = returncode
        self.stderr = stderr

    def __str__(self) -> str:
        base = super().__str__()
        extras = []
        if self.model_id:
            extras.append(f"model_id='{self.model_id}'")
        if self.executable:
            extras.append(f"executable='{self.executable}'")
        if self.returncode is not None:
            extras.append(f"returncode={self.returncode}")
        if self.stderr:
            clean_stderr = self.stderr.strip().replace("\r\n", " ").replace("\n", " ")
            if len(clean_stderr) > 120:
                clean_stderr = clean_stderr[:117] + "..."
            extras.append(f"stderr='{clean_stderr}'")
        if extras:
            return f"{base} ({', '.join(extras)})"
        return base


class RuntimeStartError(RuntimeAdapterError):
    """Raised when a runtime process fails to start or terminates prematurely on launch."""


class RuntimeProcessError(RuntimeAdapterError):
    """Raised when an operation is invalid for current process state."""


class RuntimeInferenceError(RuntimeAdapterError):
    """Raised when an inference generation attempt fails."""


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters controlling one inference generation call."""

    max_new_tokens: int = 512
    temperature: float = 0.8
    timeout: float = 120.0  # seconds before generation is killed
    stop_sequences: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True)
class InferenceResult:
    """Structured result from a single inference generation call."""

    text: str
    model_id: str
    success: bool
    elapsed_seconds: float
    prompt_tokens_estimate: int = 0
    generated_tokens_estimate: int = 0
    error: Optional[str] = None
    raw_stderr: Optional[str] = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration options for a runtime adapter."""

    executable: str = "llama-cli"
    embedding_executable: str = "llama-server"
    server_port: int = 8089
    device_policy: DevicePolicy = DevicePolicy.CPU_ONLY
    ngl: int = 0
    threads: Optional[int] = None
    startup_timeout: float = 10.0
    stop_timeout: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.device_policy, str) and not isinstance(self.device_policy, DevicePolicy):
            object.__setattr__(self, "device_policy", DevicePolicy(self.device_policy))


class RuntimeAdapter(ABC):
    """Abstract interface defining the model execution runtime contract."""

    @property
    @abstractmethod
    def state(self) -> RuntimeState:
        """Return current lifecycle state."""

    @property
    @abstractmethod
    def current_model(self) -> Optional[ModelDefinition]:
        """Return currently active model definition, if any."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check whether the runtime process is currently running."""

    @abstractmethod
    def build_command(self, model: ModelDefinition) -> List[str]:
        """Build argument command list for starting the model."""

    @abstractmethod
    def start(self, model: ModelDefinition, restart_if_running: bool = False) -> None:
        """Start the model subprocess."""

    @abstractmethod
    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop and clean up the model subprocess."""

    @abstractmethod
    def generate(
        self,
        model: ModelDefinition,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> InferenceResult:
        """Run a one-shot inference call for the given prompt and model.

        This runs a separate short-lived process distinct from the managed
        lifecycle sentinel process. It does not affect the runtime state.
        """

    @abstractmethod
    def embed(
        self,
        model: ModelDefinition,
        text: str,
    ) -> List[float]:
        """Generate an embedding vector for the text using the specified embedding model."""


class LlamaCppRuntime(RuntimeAdapter):
    """Concrete runtime adapter managing a local llama.cpp subprocess.

    Isolates process invocation, argument construction, device offload policy,
    and process lifecycle from the rest of WHIS.
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config: RuntimeConfig = config or RuntimeConfig()
        self.project_root: Path = Path(project_root).resolve() if project_root else PROJECT_ROOT
        self._state: RuntimeState = RuntimeState.STOPPED
        self._current_model: Optional[ModelDefinition] = None
        self._process: Optional[subprocess.Popen] = None
        self._last_stderr: Optional[str] = None
        self._last_command: Optional[List[str]] = None

    @property
    def state(self) -> RuntimeState:
        """Return the current lifecycle state."""
        if self._state == RuntimeState.RUNNING:
            # Query process status to synchronize state
            self.is_running()
        return self._state

    @property
    def current_model(self) -> Optional[ModelDefinition]:
        """Return the currently assigned model definition."""
        return self._current_model

    @property
    def process(self) -> Optional[subprocess.Popen]:
        """Return the raw subprocess handle if active."""
        return self._process

    @property
    def last_stderr(self) -> Optional[str]:
        """Return captured stderr from the last operation or failure."""
        return self._last_stderr

    @property
    def last_command(self) -> Optional[List[str]]:
        """Return the last constructed command list."""
        return self._last_command

    def resolve_ngl(self) -> int:
        """Resolve the number of GPU layers to offload based on device policy.

        Returns:
            0 for CPU_ONLY or AUTO (safe for Intel UHD graphics), or configured ngl for GPU_OFFLOAD.
        """
        if self.config.device_policy == DevicePolicy.CPU_ONLY:
            return 0
        if self.config.device_policy == DevicePolicy.AUTO:
            # Conservative development hardware baseline (Intel UHD Graphics): CPU execution
            return 0
        if self.config.device_policy == DevicePolicy.GPU_OFFLOAD:
            return self.config.ngl if self.config.ngl > 0 else 99
        return 0

    def resolve_model_path(self, model: ModelDefinition) -> Path:
        """Resolve model weight file path relative to project root."""
        p = Path(model.model_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def resolve_mmproj_path(self, model: ModelDefinition) -> Optional[Path]:
        """Resolve multimodal projector path relative to project root if configured."""
        if not model.mmproj_path:
            return None
        p = Path(model.mmproj_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def build_command(self, model: ModelDefinition) -> List[str]:
        """Construct a safe argument list for invoking llama.cpp for the given model.

        Ensures all paths are resolved, multimodal projectors are attached if needed,
        and device offloading policies are strictly respected. Never returns a shell string.
        """
        resolved_model_path = self.resolve_model_path(model)

        if model.type == "embedding":
            cmd = [
                self.config.embedding_executable,
                "-m",
                str(resolved_model_path),
                "--embedding",
                "--port",
                str(self.config.server_port),
            ]
            if model.default_context:
                cmd.extend(["-c", str(model.default_context)])
            ngl = self.resolve_ngl()
            cmd.extend(["-ngl", str(ngl)])
            if self.config.threads is not None and self.config.threads > 0:
                cmd.extend(["-t", str(self.config.threads)])
            self._last_command = list(cmd)
            return cmd

        cmd: List[str] = [
            self.config.executable,
            "-m",
            str(resolved_model_path),
        ]

        # Context length configuration
        if model.default_context:
            cmd.extend(["-c", str(model.default_context)])

        # Multimodal projector flag for vision/OCR models
        if model.has_projector and model.mmproj_path:
            resolved_mmproj = self.resolve_mmproj_path(model)
            if resolved_mmproj:
                cmd.extend(["--mmproj", str(resolved_mmproj)])

        # Device offloading policy
        ngl = self.resolve_ngl()
        cmd.extend(["-ngl", str(ngl)])

        # CPU threads configuration
        if self.config.threads is not None and self.config.threads > 0:
            cmd.extend(["-t", str(self.config.threads)])

        self._last_command = list(cmd)
        return cmd

    def is_running(self) -> bool:
        """Determine whether the underlying subprocess is currently running."""
        if self._process is None:
            return False

        poll_result = self._process.poll()
        if poll_result is None:
            return True

        # Process has terminated
        if self._state == RuntimeState.RUNNING:
            self._state = RuntimeState.STOPPED if poll_result == 0 else RuntimeState.FAILED
            # Attempt to read any remaining stderr
            if self._process.stderr:
                try:
                    remaining_err = self._process.stderr.read()
                    if remaining_err:
                        self._last_stderr = remaining_err
                except Exception:
                    pass
        return False

    def start(self, model: ModelDefinition, restart_if_running: bool = False) -> None:
        """Start a llama.cpp subprocess for the specified model.

        Args:
            model: Authoritative ModelDefinition from the ModelRegistry.
            restart_if_running: If True, stop any existing process before starting.
                                If False and already running, raises RuntimeProcessError.

        Raises:
            RuntimeProcessError: If the runtime is already running and restart_if_running is False.
            RuntimeStartError: If the process fails to launch or exits immediately with an error.
        """
        if self.is_running():
            if restart_if_running:
                logger.info("Stopping currently running model '%s' before restart.", self._current_model.id if self._current_model else "unknown")
                self.stop()
            else:
                active_id = self._current_model.id if self._current_model else "unknown"
                raise RuntimeProcessError(
                    f"Runtime is already running model '{active_id}'. Stop the active process before starting a new one.",
                    model_id=model.id,
                    executable=self.config.executable,
                )

        self._state = RuntimeState.STARTING
        self._current_model = model
        self._last_stderr = None

        cmd = self.build_command(model)
        logger.debug("Starting llama.cpp process: %s", cmd)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            self._state = RuntimeState.FAILED
            self._current_model = None
            self._process = None
            raise RuntimeStartError(
                f"Executable '{self.config.executable}' not found on system PATH: {exc}",
                model_id=model.id,
                executable=self.config.executable,
                command=cmd,
            ) from exc
        except Exception as exc:
            self._state = RuntimeState.FAILED
            self._current_model = None
            self._process = None
            raise RuntimeStartError(
                f"Failed to spawn runtime process: {exc}",
                model_id=model.id,
                executable=self.config.executable,
                command=cmd,
            ) from exc

        # Check for immediate launch failure
        time.sleep(0.05)
        poll_result = self._process.poll()
        if poll_result is not None:
            stderr_out = ""
            try:
                if self._process.stderr:
                    stderr_out = self._process.stderr.read()
            except Exception:
                pass

            self._last_stderr = stderr_out
            self._state = RuntimeState.FAILED
            self._current_model = None
            proc = self._process
            self._process = None

            error_msg = f"Runtime process terminated immediately on launch with returncode {poll_result}."
            if stderr_out.strip():
                error_msg += f" Stderr: {stderr_out.strip()}"

            raise RuntimeStartError(
                error_msg,
                model_id=model.id,
                executable=self.config.executable,
                command=cmd,
                returncode=poll_result,
                stderr=stderr_out or None,
            )

        if model.type == "embedding":
            # Wait for embedding server to become ready
            ready = self._wait_for_server_ready(timeout=self.config.startup_timeout)
            if not ready:
                poll = self._process.poll()
                err = ""
                try:
                    if self._process.stderr:
                        err = self._process.stderr.read()
                except Exception:
                    pass
                self.stop()
                raise RuntimeStartError(
                    f"Embedding server failed to become ready on port {self.config.server_port}. Returncode: {poll}. Stderr: {err}",
                    model_id=model.id,
                )

        self._state = RuntimeState.RUNNING
        logger.info("Successfully started runtime for model '%s'.", model.id)

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop and clean up the model subprocess cleanly.

        Sends SIGTERM first, waiting up to timeout seconds. If the process does
        not exit, sends SIGKILL. All standard streams are closed.
        """
        if self._process is None or not self.is_running():
            self._state = RuntimeState.STOPPED
            self._current_model = None
            self._process = None
            return

        self._state = RuntimeState.STOPPING
        effective_timeout = timeout if timeout is not None else self.config.stop_timeout

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not exit within %ss timeout. Killing.", effective_timeout)
                self._process.kill()
                self._process.wait(timeout=2.0)
        except Exception as exc:
            logger.error("Error while terminating process: %s", exc)
        finally:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
            self._state = RuntimeState.STOPPED
            self._current_model = None
            self._process = None
            logger.info("Runtime process stopped successfully.")

    def _build_generate_command(
        self,
        model: ModelDefinition,
        prompt: str,
        config: GenerationConfig,
    ) -> List[str]:
        """Build argument list for a one-shot inference call."""
        resolved_model_path = self.resolve_model_path(model)

        cmd: List[str] = [
            self.config.executable,
            "-m", str(resolved_model_path),
            "-p", prompt,
            "-n", str(config.max_new_tokens),
            "--temp", str(config.temperature),
            "-c", str(model.default_context),
            "-ngl", str(self.resolve_ngl()),
            "--no-display-prompt",  # suppress echoing the prompt to stdout
            "--log-disable",        # suppress llama.cpp INFO log lines
            "--simple-io",          # clean stdout for subprocess capture
            "--single-turn",        # execute single turn and exit immediately
        ]

        if model.has_projector and model.mmproj_path:
            resolved_mmproj = self.resolve_mmproj_path(model)
            if resolved_mmproj:
                cmd.extend(["--mmproj", str(resolved_mmproj)])

        if self.config.threads is not None and self.config.threads > 0:
            cmd.extend(["-t", str(self.config.threads)])

        if config.stop_sequences:
            for seq in config.stop_sequences:
                cmd.extend(["-r", seq])

        return cmd

    @staticmethod
    def _clean_output(raw_stdout: str) -> str:
        """Strip llama-cli ASCII banner, prompt echo, and trailing stats from stdout."""
        text = raw_stdout
        # Find start of conversation turn if banner was printed
        if "\n> " in text:
            text = text.split("\n> ", 1)[1]
            # Skip the prompt line itself
            if "\n" in text:
                text = text.split("\n", 1)[1]
            else:
                text = ""

        # Filter out trailing timing/status lines
        lines: List[str] = []
        for line in text.splitlines():
            trimmed = line.strip()
            if trimmed.startswith("[ Prompt:") or trimmed == "Exiting...":
                continue
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        if cleaned.startswith("Assistant:"):
            cleaned = cleaned[len("Assistant:") :].strip()
        return cleaned

    def generate(
        self,
        model: ModelDefinition,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> InferenceResult:
        """Run a one-shot inference call through llama-cli for the given prompt.

        Spawns a separate short-lived process that does not affect the managed
        lifecycle sentinel process. Process ownership and shutdown remain with
        start()/stop(). The current runtime state is not mutated.

        Args:
            model: ModelDefinition to run inference against.
            prompt: Fully constructed prompt string.
            config: GenerationConfig controlling generation parameters.

        Returns:
            InferenceResult with generated text, timing, and success status.

        Raises:
            RuntimeInferenceError: If the process fails to launch or returns an error.
        """
        cfg = config or GenerationConfig()
        cmd = self._build_generate_command(model, prompt, cfg)
        logger.debug("Generating with llama.cpp command: %s", cmd)

        t_start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t_start
            raise RuntimeInferenceError(
                f"Inference timed out after {cfg.timeout}s for model '{model.id}'.",
                model_id=model.id,
            ) from exc
        except FileNotFoundError as exc:
            elapsed = time.monotonic() - t_start
            raise RuntimeInferenceError(
                f"Executable '{self.config.executable}' not found. Cannot run inference.",
                model_id=model.id,
                executable=self.config.executable,
            ) from exc
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            raise RuntimeInferenceError(
                f"Inference failed for model '{model.id}': {exc}",
                model_id=model.id,
            ) from exc

        elapsed = time.monotonic() - t_start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # llama-cli returns 0 on success; non-zero indicates an error
        if proc.returncode != 0 and not stdout.strip():
            raise RuntimeInferenceError(
                f"Inference process exited with code {proc.returncode} for model '{model.id}'.",
                model_id=model.id,
                returncode=proc.returncode,
                stderr=stderr,
            )

        # Clean llama-cli banner and timing lines if present
        generated_text = self._clean_output(stdout)

        # Rough token estimates (chars / 4)
        prompt_tokens_est = max(1, int(len(prompt) / 4))
        generated_tokens_est = max(0, int(len(generated_text) / 4))

        logger.info(
            "Inference complete for model '%s': %.2fs, ~%d generated tokens.",
            model.id,
            elapsed,
            generated_tokens_est,
        )

        return InferenceResult(
            text=generated_text,
            model_id=model.id,
            success=True,
            elapsed_seconds=elapsed,
            prompt_tokens_estimate=prompt_tokens_est,
            generated_tokens_estimate=generated_tokens_est,
            raw_stderr=stderr if stderr.strip() else None,
        )

    def _wait_for_server_ready(self, timeout: float = 10.0) -> bool:
        """Poll the embedding server health endpoint until ready or timeout."""
        start_time = time.monotonic()
        health_url = f"http://127.0.0.1:{self.config.server_port}/health"
        while time.monotonic() - start_time < timeout:
            if self._process and self._process.poll() is not None:
                return False
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def embed(
        self,
        model: ModelDefinition,
        text: str,
        timeout: float = 30.0,
    ) -> List[float]:
        """Request an embedding vector from the running llama-server embedding process."""
        if not self.is_running():
            raise RuntimeProcessError(
                "Runtime is not running. Model must be loaded before calling embed().",
                model_id=model.id,
            )
        if self._current_model is None or self._current_model.id != model.id:
            active_id = self._current_model.id if self._current_model else "none"
            raise RuntimeProcessError(
                f"Active model '{active_id}' does not match requested model '{model.id}'.",
                model_id=model.id,
            )

        url = f"http://127.0.0.1:{self.config.server_port}/v1/embeddings"
        payload = json.dumps({"input": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["data"][0]["embedding"]
        except Exception as exc:
            raise RuntimeInferenceError(
                f"Failed to generate embedding for model '{model.id}': {exc}",
                model_id=model.id,
            ) from exc
