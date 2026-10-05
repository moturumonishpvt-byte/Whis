"""Permission model and enforcement for WHIS tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PermissionLevel(str, Enum):
    """Permission risk tiers for tool operations."""

    SAFE = "SAFE"
    CONFIRM = "CONFIRM"
    HIGH_RISK = "HIGH_RISK"

    @classmethod
    def from_str(cls, val: str | PermissionLevel) -> PermissionLevel:
        if isinstance(val, PermissionLevel):
            return val
        try:
            return cls(val.strip().upper())
        except Exception as exc:
            raise ValueError(f"Invalid permission level: {val}") from exc


@dataclass(frozen=True)
class PermissionDecision:
    """Outcome of a permission evaluation for a tool action."""

    allowed: bool
    level: PermissionLevel
    requires_confirmation: bool
    reason: Optional[str] = None


class PermissionManager:
    """Evaluates and enforces permission policies on tool operations."""

    def __init__(self, allow_high_risk: bool = False) -> None:
        self.allow_high_risk: bool = allow_high_risk

    def evaluate(
        self,
        level: PermissionLevel | str,
        confirmed: bool = False,
    ) -> PermissionDecision:
        """Evaluate whether an action at the specified permission level can execute.

        Rules:
        - SAFE: Always allowed immediately.
        - CONFIRM: Allowed if confirmed=True. If not confirmed, returns allowed=False
          with requires_confirmation=True.
        - HIGH_RISK: Never allowed automatically. Requires confirmed=True. If confirmed=True,
          it is allowed if allow_high_risk is enabled or explicit confirmation is passed.
        """
        lvl = PermissionLevel.from_str(level)

        if lvl == PermissionLevel.SAFE:
            return PermissionDecision(
                allowed=True,
                level=lvl,
                requires_confirmation=False,
            )

        if lvl == PermissionLevel.CONFIRM:
            if confirmed:
                return PermissionDecision(
                    allowed=True,
                    level=lvl,
                    requires_confirmation=False,
                )
            return PermissionDecision(
                allowed=False,
                level=lvl,
                requires_confirmation=True,
                reason="Operation requires confirmation before execution.",
            )

        if lvl == PermissionLevel.HIGH_RISK:
            if not confirmed:
                return PermissionDecision(
                    allowed=False,
                    level=lvl,
                    requires_confirmation=True,
                    reason="High-risk operation blocked. Explicit high-risk confirmation required.",
                )
            return PermissionDecision(
                allowed=True,
                level=lvl,
                requires_confirmation=False,
            )

        return PermissionDecision(
            allowed=False,
            level=lvl,
            requires_confirmation=True,
            reason=f"Unknown permission tier: {lvl}",
        )
