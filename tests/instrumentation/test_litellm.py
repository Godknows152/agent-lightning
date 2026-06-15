"""Tests for LiteLLM token ID instrumentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentlightning.instrumentation.litellm import _response_token_ids


def test_response_token_ids_supports_provider_specific_vllm_layout() -> None:
    response: dict[str, Any] = {
        "choices": [
            {
                "provider_specific_fields": {
                    "token_ids": [101, 102, 103],
                }
            }
        ]
    }

    assert _response_token_ids(response) == [101, 102, 103]


def test_response_token_ids_prefers_legacy_top_level_layout() -> None:
    response: dict[str, Any] = {
        "response_token_ids": [[201, 202]],
        "choices": [{"provider_specific_fields": {"token_ids": [301, 302]}}],
    }

    assert _response_token_ids(response) == [201, 202]


def test_response_token_ids_supports_model_like_litellm_layout() -> None:
    response = SimpleNamespace(
        response_token_ids=None,
        choices=[
            SimpleNamespace(
                token_ids=None,
                provider_specific_fields=SimpleNamespace(token_ids=[401, 402]),
            )
        ],
    )

    assert _response_token_ids(response) == [401, 402]
