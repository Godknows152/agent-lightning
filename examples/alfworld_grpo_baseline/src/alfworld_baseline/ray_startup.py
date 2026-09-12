"""Process-local Ray startup compatibility for ALFWorld.

Ray 2.54's dashboard agent binds the node IP first, then the same port on
127.0.0.1. An ephemeral port chosen on only the first address can conflict
with an existing loopback service. Probe INADDR_ANY before launching raylet.
No installed Ray files, global environment, or existing services are changed.
"""
from __future__ import annotations

import functools
import inspect
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


def select_agent_port(requested_port: int = 0) -> int:
    """Select/check an IPv4 TCP port free across all local IPv4 interfaces.

    Do not enable SO_REUSEADDR/SO_REUSEPORT: a loopback-only listener must
    exclude the candidate. The socket is released before Ray binds; as with
    other subprocess port handoffs, this cannot eliminate a concurrent bind
    occurring after the check. Explicit occupied ports fail rather than being
    silently replaced. This socket never listens or accepts connections.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        try:
            reservation.bind(("0.0.0.0", requested_port))
        except OSError as exc:
            raise RuntimeError(
                f"ALFWorld Ray dashboard agent port {requested_port} cannot be reserved "
                "across local IPv4 interfaces. Choose another port or use automatic "
                "allocation (0); no existing service has been stopped."
            ) from exc
        return int(reservation.getsockname()[1])


def install_ray_agent_port_guard(services: Any = None) -> None:
    """Guard local raylet launches in this process only; safe to call twice.

    Installed by the ALFWorld driver, not by general VERL or LLaMAFactory.
    Attaching to an existing cluster does not start a raylet and is unaffected.
    Ray's private start_raylet signature is checked to fail clearly on changes.
    """
    if services is None:
        from ray._private import services

    original = services.start_raylet
    if getattr(original, "_alfworld_agent_port_guard", False):
        return
    signature = inspect.signature(original)
    if "metrics_agent_port" not in signature.parameters:
        raise RuntimeError("Unsupported Ray start_raylet API: missing metrics_agent_port")

    @functools.wraps(original)
    def guarded_start_raylet(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        requested = bound.arguments.get("metrics_agent_port") or 0
        selected = select_agent_port(requested)
        bound.arguments["metrics_agent_port"] = selected
        logger.warning(
            "ALFWorld Ray dashboard agent gRPC port=%s (requested=%s; "
            "checked on all IPv4 interfaces, including 127.0.0.1)",
            selected,
            requested,
        )
        return original(*bound.args, **bound.kwargs)

    guarded_start_raylet._alfworld_agent_port_guard = True
    services.start_raylet = guarded_start_raylet
