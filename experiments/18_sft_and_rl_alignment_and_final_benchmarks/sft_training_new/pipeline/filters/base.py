"""
BaseFilter ABC — every Stage 2 filter implements this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """
    A single cleaning/filtering step.

    ``filter`` is called once per record.
    Returns ``(keep, reason)`` where *reason* is non-empty only when *keep* is False.
    """

    @abstractmethod
    def filter(self, record: dict) -> tuple[bool, str]:
        """Return (keep, reject_reason). reject_reason is empty string when keep=True."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used in funnel reports (e.g. 'length_filter')."""
