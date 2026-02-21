"""Compatibility helpers for cross-version runtime support."""

from __future__ import annotations

from enum import Enum

__all__ = ["StrEnum"]


class StrEnum(str, Enum):
    """Backport-compatible ``StrEnum`` replacement."""
