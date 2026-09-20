"""Compatibility export for the GiGPO-aligned ALFWorld prompt profile."""

from .prompts_gigpo import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    NONTHINKING_PROMPT_VERSION,
    PROMPT_VERSION,
    QWEN3_ALFWORLD_CHAT_TEMPLATE,
    QWEN3_XML_CALL_FORMAT,
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
)

__all__ = [
    "ALFWORLD_TEMPLATE", "ALFWORLD_TEMPLATE_NO_HIS", "NONTHINKING_PROMPT_VERSION",
    "PROMPT_VERSION", "QWEN3_ALFWORLD_CHAT_TEMPLATE", "QWEN3_XML_CALL_FORMAT",
    "SYSTEM_PROMPT", "build_messages", "build_user_prompt",
]
