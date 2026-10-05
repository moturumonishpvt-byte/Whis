"""WHIS system and productivity tools subsystem."""

from app.tools.applications import LaunchApplicationTool
from app.tools.base import BaseTool, FunctionalTool, ToolRegistry, ToolResult
from app.tools.browser import OpenURLTool
from app.tools.command import ExecuteCommandTool
from app.tools.files import (
    CreateDirectoryTool,
    DeleteFileTool,
    ListFilesTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from app.tools.system import (
    GetCurrentTimeTool,
    GetRunningApplicationsTool,
    GetSystemInfoTool,
)


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-populated with standard WHIS tools."""
    registry = ToolRegistry()

    # System tools (SAFE)
    registry.register(GetSystemInfoTool())
    registry.register(GetCurrentTimeTool())
    registry.register(GetRunningApplicationsTool())

    # File tools (SAFE, CONFIRM, HIGH_RISK)
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(CreateDirectoryTool())
    registry.register(MoveFileTool())
    registry.register(DeleteFileTool())

    # Application & Browser tools
    registry.register(LaunchApplicationTool())
    registry.register(OpenURLTool())

    # Isolated command execution (HIGH_RISK)
    registry.register(ExecuteCommandTool())

    return registry


__all__ = [
    "BaseTool",
    "CreateDirectoryTool",
    "DeleteFileTool",
    "ExecuteCommandTool",
    "FunctionalTool",
    "GetCurrentTimeTool",
    "GetRunningApplicationsTool",
    "GetSystemInfoTool",
    "LaunchApplicationTool",
    "ListFilesTool",
    "MoveFileTool",
    "OpenURLTool",
    "ReadFileTool",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "create_default_registry",
]
