# Copyright (c) Microsoft. All rights reserved.

"""LiteLLM instrumentations.

It's unclear whether or not this file is useful.
It seems that LiteLLM owns its own telemetry from their own entrance

[Related documentation](https://docs.litellm.ai/docs/observability/agentops_integration).
"""

from typing import Any, Optional

from litellm.integrations.opentelemetry import OpenTelemetry

__all__ = [
    "instrument_litellm",
    "uninstrument_litellm",
]

original_set_attributes = OpenTelemetry.set_attributes  # type: ignore


def _field(value: Any, name: str) -> Any:
    """Read one field from mapping-like or model-like LiteLLM values."""

    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _response_token_ids(response_obj: Any) -> list[int]:
    """Extract response token IDs from supported vLLM response layouts."""

    response_token_ids = _field(response_obj, "response_token_ids")
    if isinstance(response_token_ids, list) and response_token_ids:
        first_response = response_token_ids[0]
        if isinstance(first_response, list) and all(isinstance(item, int) for item in first_response):
            return first_response

    choices = _field(response_obj, "choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    token_ids = _field(first_choice, "token_ids")
    if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
        return token_ids

    provider_fields = _field(first_choice, "provider_specific_fields")
    provider_token_ids = _field(provider_fields, "token_ids")
    if isinstance(provider_token_ids, list) and all(isinstance(item, int) for item in provider_token_ids):
        return provider_token_ids
    return []


def patched_set_attributes(self: Any, span: Any, kwargs: Any, response_obj: Optional[Any]):
    original_set_attributes(self, span, kwargs, response_obj)
    # Add custom attributes
    if response_obj is not None:
        prompt_token_ids = _field(response_obj, "prompt_token_ids")
        if prompt_token_ids:
            span.set_attribute("prompt_token_ids", list(prompt_token_ids))
        response_token_ids = _response_token_ids(response_obj)
        if response_token_ids:
            span.set_attribute("response_token_ids", response_token_ids)


def instrument_litellm():
    """Instrument litellm to capture token IDs."""
    OpenTelemetry.set_attributes = patched_set_attributes


def uninstrument_litellm():
    """Uninstrument litellm to stop capturing token IDs."""
    OpenTelemetry.set_attributes = original_set_attributes
