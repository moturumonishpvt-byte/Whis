"""System information and environment inspection tools for WHIS."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
from typing import Any, Dict, List, Optional

import psutil

from app.core.permissions import PermissionLevel
from app.tools.base import BaseTool, ToolResult


class GetSystemInfoTool(BaseTool):
    """Retrieves basic system and hardware specifications."""

    name = "get_system_info"
    description = "Get basic OS, CPU, and RAM hardware metrics."
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            mem = psutil.virtual_memory()
            info = {
                "os": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "ram_total_gb": round(mem.total / (1024**3), 2),
                "ram_available_gb": round(mem.available / (1024**3), 2),
                "ram_percent_used": mem.percent,
            }
            return ToolResult.ok(self.name, output=info)
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to retrieve system info: {exc}")


class GetCurrentTimeTool(BaseTool):
    """Returns current local and UTC date and time."""

    name = "get_current_time"
    description = "Get current local and UTC date, time, and timezone."
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            now_local = datetime.now()
            now_utc = datetime.now(timezone.utc)
            data = {
                "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                "local_iso": now_local.astimezone().isoformat(),
                "utc_iso": now_utc.isoformat(),
                "timezone": str(now_local.astimezone().tzinfo),
            }
            return ToolResult.ok(self.name, output=data)
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to get current time: {exc}")


class GetRunningApplicationsTool(BaseTool):
    """Lists running process applications."""

    name = "get_running_applications"
    description = "List currently running processes and applications."
    permission_level = PermissionLevel.SAFE

    def execute(self, limit: int = 50, filter_name: Optional[str] = None, **kwargs: Any) -> ToolResult:
        try:
            processes: List[Dict[str, Any]] = []
            seen_names = set()

            for proc in psutil.process_iter(["pid", "name", "username"]):
                try:
                    pinfo = proc.info
                    pname = pinfo.get("name") or "unknown"
                    if filter_name and filter_name.lower() not in pname.lower():
                        continue
                    if pname not in seen_names:
                        seen_names.add(pname)
                        processes.append({
                            "name": pname,
                            "pid": pinfo.get("pid"),
                        })
                    if len(processes) >= limit:
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return ToolResult.ok(self.name, output=processes)
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to list running applications: {exc}")
