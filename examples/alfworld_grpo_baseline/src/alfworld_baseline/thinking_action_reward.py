"""Exact action mentions in generated thinking, without rewriting policy tokens."""

from __future__ import annotations

import re


def count_thinking_action_mentions(response: str, action: str | None) -> int:
    """Count complete, case-sensitive action occurrences before the thinking close.

    The opening thinking tag normally belongs to the prompt. The final XML call
    is excluded. Word boundaries prevent ``look`` matching ``looking`` or a
    command ending in ``1`` matching the corresponding command ending in ``10``.
    This is a literal occurrence condition, not a semantic commitment classifier.
    """
    close = response.find("</think>")
    if close < 0 or not action:
        return 0
    opening = response.find("<think>", 0, close)
    start = opening + len("<think>") if opening >= 0 else 0
    return len(re.findall(r"(?<![\w])" + re.escape(action) + r"(?![\w])", response[start:close]))
