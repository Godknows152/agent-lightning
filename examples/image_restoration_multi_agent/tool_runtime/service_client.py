"""Small standard-library client for the persistent local tool runtime."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, cast


def post_json(base_url: str, endpoint: str, payload: dict[str, object], timeout_seconds: float) -> dict[str, Any]:
    """POST one JSON request and return a validated object response."""

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"persistent tool service returned HTTP {error.code}: {detail[-2000:]}") from error
    if not isinstance(response_payload, dict):
        raise ValueError("persistent tool service response must be a JSON object")
    return cast(dict[str, Any], response_payload)
