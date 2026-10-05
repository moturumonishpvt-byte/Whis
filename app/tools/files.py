"""File system inspection and manipulation tools for WHIS."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from app.core.permissions import PermissionLevel
from app.tools.base import BaseTool, ToolResult


def _validate_path(path_str: str) -> Path:
    """Validate and resolve path, checking for empty strings or invalid characters."""
    if not path_str or not str(path_str).strip():
        raise ValueError("Path cannot be empty.")
    if "\x00" in path_str:
        raise ValueError("Null bytes are not allowed in paths.")
    return Path(path_str).resolve()


class ListFilesTool(BaseTool):
    """Lists files and subdirectories in a directory."""

    name = "list_files"
    description = "List files and directories in a given path."
    permission_level = PermissionLevel.SAFE

    def execute(self, path: str = ".", max_items: int = 100, **kwargs: Any) -> ToolResult:
        try:
            target = _validate_path(path)
            if not target.exists():
                return ToolResult.fail(self.name, f"Path '{path}' does not exist.")
            if not target.is_dir():
                return ToolResult.fail(self.name, f"Path '{path}' is not a directory.")

            items: List[Dict[str, Any]] = []
            for entry in target.iterdir():
                try:
                    stat = entry.stat()
                    items.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size_bytes": stat.st_size if entry.is_file() else 0,
                        "modified": stat.st_mtime,
                    })
                except (PermissionError, OSError):
                    items.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size_bytes": 0,
                        "accessible": False,
                    })
                if len(items) >= max_items:
                    break

            return ToolResult.ok(self.name, output={
                "directory": str(target),
                "items": items,
                "total_items_reported": len(items),
            })
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to list directory '{path}': {exc}")


class ReadFileTool(BaseTool):
    """Reads content from a text file."""

    name = "read_file"
    description = "Read text content from a file."
    permission_level = PermissionLevel.SAFE

    def execute(self, path: str, max_chars: int = 50000, **kwargs: Any) -> ToolResult:
        try:
            target = _validate_path(path)
            if not target.exists():
                return ToolResult.fail(self.name, f"File '{path}' does not exist.")
            if not target.is_file():
                return ToolResult.fail(self.name, f"Path '{path}' is not a regular file.")

            content = target.read_text(encoding="utf-8", errors="replace")
            truncated = len(content) > max_chars
            return ToolResult.ok(self.name, output={
                "path": str(target),
                "content": content[:max_chars],
                "char_count": len(content),
                "truncated": truncated,
            })
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to read file '{path}': {exc}")


class WriteFileTool(BaseTool):
    """Writes text content to a file."""

    name = "write_file"
    description = "Write text content to a file (requires confirmation)."
    permission_level = PermissionLevel.CONFIRM

    def execute(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            target = _validate_path(path)
            if target.exists() and not overwrite:
                return ToolResult.fail(
                    self.name,
                    f"File '{path}' already exists and overwrite is False.",
                )

            # Ensure parent directories exist
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

            return ToolResult.ok(self.name, output={
                "path": str(target),
                "bytes_written": len(content.encode("utf-8")),
            })
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to write file '{path}': {exc}")


class CreateDirectoryTool(BaseTool):
    """Creates a directory at the given path."""

    name = "create_directory"
    description = "Create a new directory (requires confirmation)."
    permission_level = PermissionLevel.CONFIRM

    def execute(self, path: str, exist_ok: bool = True, **kwargs: Any) -> ToolResult:
        try:
            target = _validate_path(path)
            target.mkdir(parents=True, exist_ok=exist_ok)
            return ToolResult.ok(self.name, output={"path": str(target), "created": True})
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to create directory '{path}': {exc}")


class MoveFileTool(BaseTool):
    """Moves or renames a file from source to destination."""

    name = "move_file"
    description = "Move or rename a file (requires confirmation)."
    permission_level = PermissionLevel.CONFIRM

    def execute(self, source: str, destination: str, **kwargs: Any) -> ToolResult:
        try:
            src = _validate_path(source)
            dst = _validate_path(destination)

            if not src.exists():
                return ToolResult.fail(self.name, f"Source path '{source}' does not exist.")

            # If destination is an existing directory, move into it
            if dst.is_dir():
                dst = dst / src.name

            shutil.move(str(src), str(dst))
            return ToolResult.ok(self.name, output={
                "source": str(src),
                "destination": str(dst),
                "moved": True,
            })
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to move '{source}' to '{destination}': {exc}")


class DeleteFileTool(BaseTool):
    """Deletes a single file (requires explicit HIGH_RISK confirmation)."""

    name = "delete_file"
    description = "Delete a single file (HIGH RISK: requires explicit confirmation)."
    permission_level = PermissionLevel.HIGH_RISK

    def execute(self, path: str, **kwargs: Any) -> ToolResult:
        try:
            target = _validate_path(path)
            if not target.exists():
                return ToolResult.fail(self.name, f"File '{path}' does not exist.")
            if target.is_dir():
                return ToolResult.fail(
                    self.name,
                    f"Target '{path}' is a directory. Recursive destructive deletion is forbidden.",
                )

            target.unlink()
            return ToolResult.ok(self.name, output={"path": str(target), "deleted": True})
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to delete file '{path}': {exc}")
