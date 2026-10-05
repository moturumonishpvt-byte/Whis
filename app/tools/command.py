"""Isolated high-risk command execution facility for WHIS."""

from __future__ import annotations

import subprocess
from typing import Any, List, Union

from app.core.permissions import PermissionLevel
from app.tools.base import BaseTool, ToolResult


class ExecuteCommandTool(BaseTool):
    """Executes a local command process using discrete arguments without shell=True.

    HIGH_RISK: Never executes automatically. Requires explicit high-risk confirmation.
    """

    name = "execute_command"
    description = "Execute a local command with discrete arguments (HIGH RISK: requires explicit confirmation)."
    permission_level = PermissionLevel.HIGH_RISK

    def execute(
        self,
        command: Union[str, List[str]],
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> ToolResult:
        if not command:
            return ToolResult.fail(self.name, "Command cannot be empty.")

        if isinstance(command, str):
            import shlex
            args = shlex.split(command, posix=False)
        elif isinstance(command, list):
            args = [str(a) for a in command]
        else:
            return ToolResult.fail(self.name, "Command must be a list of strings or argument string.")

        if not args:
            return ToolResult.fail(self.name, "Empty argument list.")

        try:
            res = subprocess.run(
                args,
                shell=False,  # strictly enforce shell=False
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ToolResult.ok(self.name, output={
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "command": args,
            })
        except subprocess.TimeoutExpired:
            return ToolResult.fail(self.name, f"Command timed out after {timeout} seconds.")
        except Exception as exc:
            return ToolResult.fail(self.name, f"Command execution failed: {exc}")
