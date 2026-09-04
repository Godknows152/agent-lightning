"""Compatibility export for callers that still expect the Qwen2.5 profile."""

from .prompts_qwen25 import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, build_user_prompt

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_messages", "build_user_prompt"]
