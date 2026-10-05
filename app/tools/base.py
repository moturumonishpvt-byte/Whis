"""Base classes and execution registry for WHIS tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional

from app.core.permissions import PermissionDecision, PermissionLevel, PermissionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """Standardized result returned by tool execution."""

    success: bool
    tool_name: str
    output: Any = None
    error: Optional[str] = None
    requires_confirmation: bool = False

    @classmethod
    def ok(cls, tool_name: str, output: Any = None) -> ToolResult:
        """Create a successful tool result."""
        return cls(success=True, tool_name=tool_name, output=output)

    @classmethod
    def fail(
        cls,
        tool_name: str,
        error: str,
        requires_confirmation: bool = False,
    ) -> ToolResult:
        """Create a failed tool result."""
        return cls(
            success=False,
            tool_name=tool_name,
            error=error,
            requires_confirmation=requires_confirmation,
        )


class BaseTool(ABC):
    """Abstract base class for all WHIS tools."""

    name: str
    description: str
    permission_level: PermissionLevel = PermissionLevel.SAFE

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with supplied arguments. Must return a ToolResult."""


class FunctionalTool(BaseTool):
    """Wraps a Python callable into a BaseTool instance."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        permission_level: PermissionLevel = PermissionLevel.SAFE,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.permission_level = permission_level

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            res = self.func(**kwargs)
            if isinstance(res, ToolResult):
                return res
            return ToolResult.ok(self.name, output=res)
        except Exception as exc:
            logger.error("Error executing functional tool '%s': %s", self.name, exc)
            return ToolResult.fail(self.name, str(exc))


class ToolRegistry:
    """Registry and executor for WHIS tools with permission checking."""

    def __init__(self, permission_manager: Optional[PermissionManager] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self.permissions: PermissionManager = (
            permission_manager if permission_manager is not None else PermissionManager()
        )

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name or not tool.name.strip():
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.permission_level.value)

    def get(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "permission_level": t.permission_level.value,
            }
            for t in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        confirmed: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a registered tool by name with permission verification.

        Args:
            name: Tool name.
            confirmed: Explicit confirmation flag for CONFIRM / HIGH_RISK tools.
            **kwargs: Arguments passed to the tool.

        Returns:
            ToolResult describing execution success or permission blockage.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(name, f"Tool '{name}' not found.")

        # Check permissions
        decision: PermissionDecision = self.permissions.evaluate(
            tool.permission_level,
            confirmed=confirmed,
        )

        if not decision.allowed:
            return ToolResult.fail(
                name,
                decision.reason or "Permission denied.",
                requires_confirmation=decision.requires_confirmation,
            )

        # Execute safely, never leaking raw unhandled exceptions
        try:
            return tool.execute(**kwargs)
        except Exception as exc:
            logger.error("Unhandled exception during tool execution '%s': %s", name, exc)
            return ToolResult.fail(name, f"Execution failed: {exc}")
