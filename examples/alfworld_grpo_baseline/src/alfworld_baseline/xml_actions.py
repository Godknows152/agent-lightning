"""Strict v7 XML decisions; parsing never rewrites sampled tokens or log-probs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .thinking import tool_output


@dataclass(frozen=True)
class XMLDecision:
    status: Literal["valid", "no_action", "invalid_action"]
    reason: str
    action: str | None = None


def parse_xml_decision(text: str, *, enable_thinking: bool = True) -> XMLDecision:
    """Separate absent/incomplete calls from complete but invalid schema output.

    Thinking is not executable. Only terminal transport markers are removed
    from the parser view. Multiple calls or surrounding prose are rejected,
    never repaired or silently trimmed to the first call.
    """
    output, _ = tool_output(text, enable_thinking=enable_thinking)
    if enable_thinking and "</think>" not in text:
        return XMLDecision("no_action", "unclosed_thinking")
    output = output.strip()
    while output.endswith(("<|im_end|>", "<|endoftext|>")):
        output = output[:output.rfind("<|")].rstrip()
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not calls:
        return XMLDecision("no_action", "missing_or_incomplete_call")
    if len(calls) != 1 or output.count("<tool_call>") != 1 or output.count("</tool_call>") != 1:
        return XMLDecision("invalid_action", "multiple_calls")
    if re.fullmatch(r"<tool_call>.*?</tool_call>", output, re.DOTALL) is None:
        return XMLDecision("invalid_action", "text_outside_call")
    body = calls[0].strip()
    function = re.fullmatch(r"<function=([^<>\s]+)>(.*?)</function>", body, re.DOTALL)
    if function is None:
        return XMLDecision("no_action", "malformed_xml")
    name, parameters = function.groups()
    if name != "alfworld_action":
        return XMLDecision("invalid_action", "unknown_tool")
    # Unclosed structures are not executable calls; balanced extra/missing
    # parameters are schema violations rather than successful empty actions.
    if parameters.count("<parameter=") != parameters.count("</parameter>"):
        return XMLDecision("no_action", "incomplete_parameter")
    parameter = re.fullmatch(r"\s*<parameter=action>([^<>]*)</parameter>\s*", parameters, re.DOTALL)
    if parameter is None:
        return XMLDecision("invalid_action", "invalid_arguments_schema")
    action = parameter.group(1).strip()
    if not action or len(action.splitlines()) != 1:
        return XMLDecision("invalid_action", "invalid_action_value")
    return XMLDecision("valid", "parsed", action)
