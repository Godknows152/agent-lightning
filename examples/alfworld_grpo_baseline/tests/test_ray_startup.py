"""CPU-only checks for dashboard-agent ports; never start training."""
from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from alfworld_baseline.ray_startup import install_ray_agent_port_guard, select_agent_port


def test_automatic_port_avoids_loopback_listener():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        used = occupied.getsockname()[1]
        for _ in range(32):
            selected = select_agent_port()
            assert selected != used
            with socket.socket() as checked:
                checked.bind(("0.0.0.0", selected))
        # Existing service was not stopped or modified.
        assert occupied.getsockname()[1] == used


def test_explicit_occupied_port_fails_without_replacing_it():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(RuntimeError, match="no existing service has been stopped"):
            select_agent_port(occupied.getsockname()[1])


def test_explicit_available_port_preserved():
    port = select_agent_port()
    assert select_agent_port(port) == port


@pytest.mark.parametrize("requested", [None, 0])
def test_guard_preserves_other_arguments_and_result(requested):
    def start_raylet(node_ip_address, metrics_agent_port=None, marker=None):
        return node_ip_address, metrics_agent_port, marker

    services = SimpleNamespace(start_raylet=start_raylet)
    install_ray_agent_port_guard(services)
    marker = object()
    ip, port, result = services.start_raylet("10.246.1.30", requested, marker=marker)
    assert ip == "10.246.1.30"
    assert 0 < port <= 65535
    assert result is marker


def test_guard_explicit_port_and_idempotence():
    services = SimpleNamespace(start_raylet=lambda metrics_agent_port=0: metrics_agent_port)
    install_ray_agent_port_guard(services)
    wrapper = services.start_raylet
    install_ray_agent_port_guard(services)
    assert services.start_raylet is wrapper
    port = select_agent_port()
    assert services.start_raylet(metrics_agent_port=port) == port


def test_guard_occupied_port_does_not_launch():
    def start_raylet(metrics_agent_port=0):
        pytest.fail("Ray must not start with a known conflicting port")

    services = SimpleNamespace(start_raylet=start_raylet)
    install_ray_agent_port_guard(services)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(RuntimeError):
            services.start_raylet(occupied.getsockname()[1])


def test_incompatible_ray_api_fails_clearly():
    with pytest.raises(RuntimeError, match="Unsupported Ray"):
        install_ray_agent_port_guard(SimpleNamespace(start_raylet=lambda: None))


def test_installed_ray_api_is_supported(monkeypatch):
    from ray._private import services

    original = services.start_raylet
    monkeypatch.setattr(services, "start_raylet", original)
    install_ray_agent_port_guard(services)
    assert services.start_raylet.__wrapped__ is original
