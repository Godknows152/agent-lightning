"""Separate Qwen thinking from executable output without altering training tokens."""
from __future__ import annotations

from typing import Any


def tool_output(text: str, *, enable_thinking: bool) -> tuple[str, int]:
    """Return post-thinking output and its character offset.

    The generation prompt already contains ``<think>``. Until the generated
    closing tag arrives, nothing (including XML quoted in reasoning) executes.
    Disabled thinking preserves the original strict protocol unchanged.
    """
    if not enable_thinking:
        return text, 0
    end = text.find("</think>")
    if end < 0:
        return "", len(text)
    offset = end + len("</think>")
    return text[offset:], offset


class ThinkingToolParser:
    """Filter only the parser view; never replace rollout IDs or logprobs."""

    def __init__(self, parser: Any, tokenizer: Any):
        self.parser = parser
        self.tokenizer = tokenizer

    async def extract_tool_calls(self, responses_ids: list[int], tools: Any = None):
        text = self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        output, _ = tool_output(text, enable_thinking=True)
        if not output.strip():
            return "", []
        # Retokenization is only for the downstream parser, not policy learning.
        return await self.parser.extract_tool_calls(
            self.tokenizer.encode(output, add_special_tokens=False), tools
        )
