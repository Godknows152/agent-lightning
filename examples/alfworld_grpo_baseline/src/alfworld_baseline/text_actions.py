"""Parse only post-thinking text, without changing sampled IDs or log-probabilities."""
from __future__ import annotations

from .thinking import tool_output


def parse_text_action(text: str, *, enable_thinking: bool = True) -> str | None:
    """Return a single command (optionally prefixed by ``Action:``).

    Missing/unclosed thinking, empty output, multiple lines, and structured
    tool/JSON output are no-action failures. Membership in the *current*
    admissible list is checked by the environment, not by this parser.
    Only terminal transport markers are removed. No fuzzy action repair.
    """
    output, _ = tool_output(text, enable_thinking=enable_thinking)
    output = output.strip()
    while output.endswith(("<|im_end|>", "<|endoftext|>")):
        output = output.rsplit("<|", 1)[0].rstrip()
    if output.startswith("Action:"):
        output = output[len("Action:"):].strip()
    if not output or len(output.splitlines()) != 1:
        return None
    if any(char in output for char in "<>{}`") or "Action:" in output:
        return None
    return output
