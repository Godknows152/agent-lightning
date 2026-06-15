#!/usr/bin/env python3
"""Serve persistent restoration and IQA workers from the isolated `verl` environment."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, Sequence, TextIO, cast

import yaml

RESULT_PREFIX = "RESULT_JSON="
READY_PREFIX = "READY_JSON="


class JsonlWorker:
    """One long-lived model process using prefixed JSON records over stdio."""

    def __init__(self, name: str, command: list[str]) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"},
        )
        self._stdin = cast(TextIO, self._process.stdin)
        self._stdout = cast(TextIO, self._process.stdout)
        try:
            ready = self._read_record(READY_PREFIX)
        except Exception:
            self.close()
            raise
        if ready.get("status") != "ready":
            raise RuntimeError(f"{name} failed to initialize: {ready}")
        self.ready = ready

    def _read_record(self, prefix: str) -> dict[str, Any]:
        while True:
            line = self._stdout.readline()
            if line == "":
                return_code = self._process.poll()
                raise RuntimeError(f"{self.name} exited before emitting {prefix}; return_code={return_code}")
            if line.startswith(prefix):
                payload = json.loads(line.removeprefix(prefix))
                if not isinstance(payload, dict):
                    raise ValueError(f"{self.name} emitted a non-object record")
                return cast(dict[str, Any], payload)
            print(f"[{self.name}] {line.rstrip()}", file=sys.stderr, flush=True)

    def request(self, payload: dict[str, object]) -> dict[str, Any]:
        with self._lock:
            request_id = uuid.uuid4().hex
            self._stdin.write(json.dumps({**payload, "request_id": request_id}, separators=(",", ":")) + "\n")
            self._stdin.flush()
            response = self._read_record(RESULT_PREFIX)
            if response.get("request_id") != request_id:
                raise RuntimeError(f"{self.name} returned a mismatched request_id")
            if response.get("status") != "success":
                raise RuntimeError(f"{self.name} request failed: {response.get('error', response)}")
            return response

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)


class RequestWorker(Protocol):
    """Worker interface required by the bounded request pool."""

    def request(self, payload: dict[str, object]) -> dict[str, Any]: ...


class JsonlWorkerPool:
    """Distribute concurrent requests across equivalent persistent workers."""

    def __init__(self, workers: Sequence[RequestWorker]) -> None:
        if not workers:
            raise ValueError("worker pool must contain at least one worker")
        self.workers = list(workers)
        self._available: queue.Queue[RequestWorker] = queue.Queue(maxsize=len(workers))
        for worker in workers:
            self._available.put_nowait(worker)

    def request(self, payload: dict[str, object]) -> dict[str, Any]:
        worker = self._available.get()
        try:
            return worker.request(payload)
        finally:
            self._available.put_nowait(worker)


class PersistentToolRuntime:
    """Own persistent model pools and run independent requests concurrently."""

    def __init__(
        self,
        *,
        tools_config: Path,
        external_tools_root: Path,
        iqa_repo: Path,
        restoration_devices: list[str],
        iqa_devices: list[str],
        metrics: list[str],
        restoration_workers: int = 1,
        iqa_workers: int = 1,
    ) -> None:
        if restoration_workers < 1:
            raise ValueError("restoration_workers must be positive")
        if iqa_workers < 1:
            raise ValueError("iqa_workers must be positive")
        if not restoration_devices:
            raise ValueError("restoration_devices must not be empty")
        if not iqa_devices:
            raise ValueError("iqa_devices must not be empty")
        self.restoration_devices = restoration_devices
        self.iqa_devices = iqa_devices
        self.metrics = metrics
        self.restoration_workers = restoration_workers
        self.iqa_workers = iqa_workers
        self._workers: list[JsonlWorker] = []
        self._action_workers: dict[str, tuple[JsonlWorkerPool, str]] = {}
        self._state = "starting"
        self._state_lock = threading.Lock()

        tool_payload = yaml.safe_load(tools_config.read_text(encoding="utf-8"))
        tools = tool_payload.get("tools") if isinstance(tool_payload, dict) else None
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"tools configuration has no tools: {tools_config}")

        script_dir = Path(__file__).resolve().parent
        iqa_pool_workers = [
            JsonlWorker(
                f"iqa-{worker_index}",
                [
                    sys.executable,
                    str(script_dir / "iqa_entrypoint.py"),
                    "--repo",
                    str(iqa_repo),
                    "--metrics",
                    ",".join(metrics),
                    "--device",
                    iqa_devices[worker_index % len(iqa_devices)],
                    "--serve-jsonl",
                ],
            )
            for worker_index in range(iqa_workers)
        ]
        self._workers.extend(iqa_pool_workers)
        self._iqa_pool = JsonlWorkerPool(iqa_pool_workers)

        toolkit_tools = [tool for tool in tools if tool["runtime"]["adapter"] == "verl_toolkit"]
        toolkit_models = [str(tool["runtime"]["model"]) for tool in toolkit_tools]
        toolkit_pool_workers = [
            JsonlWorker(
                f"restoration-toolkit-{worker_index}",
                [
                    sys.executable,
                    str(script_dir / "restoration_entrypoint.py"),
                    "--adapter",
                    "verl_toolkit",
                    "--models",
                    ",".join(toolkit_models),
                    "--external-tools-root",
                    str(external_tools_root),
                    "--device",
                    restoration_devices[worker_index % len(restoration_devices)],
                    "--serve-jsonl",
                ],
            )
            for worker_index in range(restoration_workers)
        ]
        self._workers.extend(toolkit_pool_workers)
        toolkit_pool = JsonlWorkerPool(toolkit_pool_workers)
        for tool in toolkit_tools:
            self._action_workers[str(tool["name"])] = (toolkit_pool, str(tool["runtime"]["model"]))

        for tool in tools:
            runtime = tool["runtime"]
            if runtime["adapter"] != "candidate":
                continue
            action = str(tool["name"])
            model = str(runtime["model"])
            action_pool_workers = [
                JsonlWorker(
                    f"restoration-{action}-{worker_index}",
                    [
                        sys.executable,
                        str(script_dir / "restoration_entrypoint.py"),
                        "--adapter",
                        "candidate",
                        "--model",
                        model,
                        "--external-tools-root",
                        str(external_tools_root),
                        "--repo",
                        str((external_tools_root / runtime["repo"]).resolve()),
                        "--checkpoint",
                        str((external_tools_root / runtime["checkpoint"]).resolve()),
                        "--device",
                        restoration_devices[worker_index % len(restoration_devices)],
                        "--serve-jsonl",
                    ],
                )
                for worker_index in range(restoration_workers)
            ]
            self._workers.extend(action_pool_workers)
            self._action_workers[action] = (JsonlWorkerPool(action_pool_workers), model)

        action_names = {str(tool["name"]) for tool in tools}
        if set(self._action_workers) != action_names:
            missing = sorted(action_names - set(self._action_workers))
            raise RuntimeError(f"persistent restoration workers are missing actions: {missing}")
        self._state = "ready"

    def _require_ready(self) -> None:
        with self._state_lock:
            if self._state != "ready":
                raise RuntimeError(f"tool runtime is not ready: state={self._state}")

    def restore(self, payload: dict[str, object]) -> dict[str, Any]:
        self._require_ready()
        action = str(payload["action"])
        worker_and_model = self._action_workers.get(action)
        if worker_and_model is None:
            raise ValueError(f"unknown restoration action: {action}")
        worker_pool, model = worker_and_model
        return worker_pool.request(
            {
                "model": model,
                "input": str(payload["input_path"]),
                "output": str(payload["output_path"]),
            }
        )

    def evaluate(self, payload: dict[str, object]) -> dict[str, Any]:
        self._require_ready()
        return self._iqa_pool.request({"input": str(payload["image_path"])})

    def sleep(self) -> dict[str, object]:
        """Unload every worker model after all rollout requests have completed."""
        with self._state_lock:
            if self._state == "sleeping":
                return {"status": "sleeping"}
            if self._state != "ready":
                raise RuntimeError(f"cannot sleep tool runtime from state={self._state}")
            self._state = "sleeping_pending"
        try:
            workers = [worker.request({"command": "sleep"}) for worker in self._workers]
        except Exception:
            with self._state_lock:
                self._state = "failed"
            raise
        with self._state_lock:
            self._state = "sleeping"
        return {"status": "sleeping", "workers": workers}

    def wake(self) -> dict[str, object]:
        """Reload every worker model before the next rollout phase."""
        with self._state_lock:
            if self._state == "ready":
                return {"status": "ready"}
            if self._state != "sleeping":
                raise RuntimeError(f"cannot wake tool runtime from state={self._state}")
            self._state = "waking"
        try:
            workers = [worker.request({"command": "wake"}) for worker in self._workers]
        except Exception:
            with self._state_lock:
                self._state = "failed"
            raise
        with self._state_lock:
            self._state = "ready"
        return {"status": "ready", "workers": workers}

    def health(self) -> dict[str, object]:
        iqa_worker_devices = [str(worker.ready["device"]) for worker in self._iqa_pool.workers]
        restoration_worker_devices = {
            action: [str(worker.ready["device"]) for worker in worker_pool.workers]
            for action, (worker_pool, _model) in self._action_workers.items()
        }
        return {
            "status": self._state,
            "restoration_devices": self.restoration_devices,
            "iqa_devices": self.iqa_devices,
            "actions": sorted(self._action_workers),
            "metrics": self.metrics,
            "restoration_workers": self.restoration_workers,
            "iqa_workers": self.iqa_workers,
            "iqa_worker_devices": iqa_worker_devices,
            "restoration_worker_devices": restoration_worker_devices,
            "workers": {worker.name: worker.ready for worker in self._workers},
        }

    def close(self) -> None:
        for worker in reversed(self._workers):
            worker.close()


def _handler(runtime: PersistentToolRuntime) -> type[BaseHTTPRequestHandler]:
    class ToolRequestHandler(BaseHTTPRequestHandler):
        server_version = "ImageRestorationToolRuntime/1"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[http] {format_string % args}", file=sys.stderr, flush=True)

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"status": "failed", "error": "unknown endpoint"})
                return
            self._write_json(HTTPStatus.OK, runtime.health())

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/restore":
                    response = runtime.restore(cast(dict[str, object], payload))
                elif self.path == "/evaluate":
                    response = runtime.evaluate(cast(dict[str, object], payload))
                elif self.path == "/sleep":
                    response = runtime.sleep()
                elif self.path == "/wake":
                    response = runtime.wake()
                else:
                    self._write_json(HTTPStatus.NOT_FOUND, {"status": "failed", "error": "unknown endpoint"})
                    return
                self._write_json(HTTPStatus.OK, cast(dict[str, object], response))
            except Exception as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                )

    return ToolRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--tools-config", type=Path, required=True)
    parser.add_argument("--external-tools-root", type=Path, required=True)
    parser.add_argument("--iqa-repo", type=Path, required=True)
    parser.add_argument("--metrics", default="maniqa,niqe,clipiqa,topiq_nr")
    parser.add_argument("--restoration-devices", default="cuda:0")
    parser.add_argument("--iqa-devices", default="cuda:0")
    parser.add_argument("--restoration-workers", type=int, default=1)
    parser.add_argument("--iqa-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = PersistentToolRuntime(
        tools_config=args.tools_config.expanduser().resolve(),
        external_tools_root=args.external_tools_root.expanduser().resolve(),
        iqa_repo=args.iqa_repo.expanduser().resolve(),
        restoration_devices=[item.strip() for item in args.restoration_devices.split(",") if item.strip()],
        iqa_devices=[item.strip() for item in args.iqa_devices.split(",") if item.strip()],
        metrics=[item.strip() for item in args.metrics.split(",") if item.strip()],
        restoration_workers=args.restoration_workers,
        iqa_workers=args.iqa_workers,
    )
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), _handler(runtime))

    def request_shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        f"Persistent tool runtime ready at http://{args.host}:{args.port}; "
        f"IQA={args.iqa_devices} x{args.iqa_workers}, "
        f"restoration={args.restoration_devices} x{args.restoration_workers}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
