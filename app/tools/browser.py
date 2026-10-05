"""Browser interaction tools for WHIS."""

from __future__ import annotations

from typing import Any
import urllib.parse
import webbrowser

from app.core.permissions import PermissionLevel
from app.tools.base import BaseTool, ToolResult

ALLOWED_URL_SCHEMES = {"http", "https"}


class OpenURLTool(BaseTool):
    """Safely opens an HTTP or HTTPS web URL in the default system browser."""

    name = "open_url"
    description = "Open a valid web URL in the default browser."
    permission_level = PermissionLevel.SAFE

    def execute(self, url: str, **kwargs: Any) -> ToolResult:
        if not url or not str(url).strip():
            return ToolResult.fail(self.name, "URL cannot be empty.")

        clean_url = str(url).strip()
        parsed = urllib.parse.urlparse(clean_url)

        if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
            return ToolResult.fail(
                self.name,
                f"Invalid URL scheme '{parsed.scheme}'. Only http and https URLs are permitted.",
            )

        if not parsed.netloc:
            return ToolResult.fail(self.name, "URL must include a valid network host domain.")

        try:
            opened = webbrowser.open(clean_url)
            return ToolResult.ok(self.name, output={"url": clean_url, "opened": opened})
        except Exception as exc:
            return ToolResult.fail(self.name, f"Failed to open URL in browser: {exc}")
