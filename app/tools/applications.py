"""Application launching tools for Windows environment."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from app.core.permissions import PermissionLevel
from app.tools.base import BaseTool, ToolResult

# Safe map of common Windows utilities to known binaries
COMMON_WINDOWS_APPS: Dict[str, str] = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "mspaint": "mspaint.exe",
    "paint": "mspaint.exe",
}

# Forbidden shell injection patterns
SHELL_METACHARS_RE = re.compile(r"[&|;><`$\n\r]")


class LaunchApplicationTool(BaseTool):
    """Launches an application or executable safely without shell=True."""

    name = "launch_application"
    description = "Launch a local Windows application or executable (requires confirmation)."
    permission_level = PermissionLevel.CONFIRM

    def execute(
        self,
        name_or_path: str,
        arguments: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not name_or_path or not str(name_or_path).strip():
            return ToolResult.fail(self.name, "Application name or path cannot be empty.")

        clean_name = str(name_or_path).strip()

        # Reject any shell metacharacters in target name
        if SHELL_METACHARS_RE.search(clean_name):
            return ToolResult.fail(
                self.name,
                "Command contains forbidden shell metacharacters.",
            )

        # Resolve binary
        executable: Optional[str] = None
        lower_name = clean_name.lower().removesuffix(".exe")
        if lower_name in COMMON_WINDOWS_APPS:
            executable = COMMON_WINDOWS_APPS[lower_name]
        else:
            # Check if it exists as an explicit file or on PATH
            p = Path(clean_name)
            if p.is_file():
                executable = str(p.resolve())
            else:
                found = shutil.which(clean_name)
                if found:
                    executable = found

        if not executable:
            return ToolResult.fail(
                self.name,
                f"Application '{clean_name}' not found on system or PATH.",
            )

        # Sanitize arguments
        arg_list: List[str] = [executable]
        if arguments:
            for arg in arguments:
                s_arg = str(arg)
                if SHELL_METACHARS_RE.search(s_arg):
                    return ToolResult.fail(
                        self.name,
                        f"Argument contains forbidden shell metacharacters: {s_arg}",
                    )
                arg_list.append(s_arg)

        # Launch without shell=True
        try:
            proc = subprocess.Popen(
                arg_list,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ToolResult.ok(self.name, output={
                "executable": executable,
                "pid": proc.pid,
                "launched": True,
            })
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to launch application '{executable}': {exc}")
