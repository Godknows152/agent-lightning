"""Unit tests for stage D subprocess adapters without loading GPU models."""

from __future__ import annotations

import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import evaluators.pyiqa_evaluator as evaluator_module
import workers.subprocess_worker as worker_module
from config import EvaluatorSettings, IQAMetricConfig, SubprocessSettings
from evaluators.pyiqa_evaluator import PyiqaSubprocessEvaluator
from PIL import Image
from schemas import ExecutionStatus
from tool_registry import ToolDefinition, ToolRegistry, ToolRegistryConfig, ToolRuntime
from tool_runtime.persistent_tool_server import JsonlWorker, JsonlWorkerPool, _handler
from tool_runtime.restoration_entrypoint import _bootstrap_basicsr_utils, _bootstrap_retinexformer_utils
from tool_runtime.service_client import post_json
from workers import SubprocessRestorationWorker


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(80, 100, 120)).save(path)


def _registry() -> ToolRegistry:
    return ToolRegistry(
        ToolRegistryConfig(
            registry_name="real-test",
            tools=[
                ToolDefinition(
                    name="test_tool",
                    description="Test subprocess tool",
                    runtime=ToolRuntime(adapter="verl_toolkit", model="test_model"),
                )
            ],
        )
    )


def test_jsonl_worker_reuses_one_long_lived_process(tmp_path: Path) -> None:
    worker_script = tmp_path / "persistent_worker.py"
    worker_script.write_text(
        "import json, os, sys\n"
        "print('READY_JSON='+json.dumps({'status':'ready','pid':os.getpid()}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    command=request.get('command','infer')\n"
        "    print('RESULT_JSON='+json.dumps({'status':'success','request_id':request['request_id'],"
        "'pid':os.getpid(),'command':command,'value':request.get('value')}), flush=True)\n",
        encoding="utf-8",
    )
    worker = JsonlWorker("test", [sys.executable, str(worker_script)])
    try:
        first = worker.request({"value": 1})
        sleeping = worker.request({"command": "sleep"})
        waking = worker.request({"command": "wake"})
        second = worker.request({"value": 2})
    finally:
        worker.close()

    assert first["pid"] == worker.ready["pid"]
    assert second["pid"] == worker.ready["pid"]
    assert (first["value"], second["value"]) == (1, 2)
    assert sleeping["command"] == "sleep"
    assert waking["command"] == "wake"


def test_jsonl_worker_pool_runs_requests_concurrently() -> None:
    barrier = threading.Barrier(2)

    class FakeWorker:
        def __init__(self, name: str) -> None:
            self.name = name

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            barrier.wait(timeout=2)
            return {"worker": self.name, **payload}

    workers = [FakeWorker("worker-0"), FakeWorker("worker-1")]
    pool = JsonlWorkerPool(workers)
    results: list[dict[str, object]] = []

    def request(value: int) -> None:
        results.append(pool.request({"value": value}))

    threads = [threading.Thread(target=request, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert {result["worker"] for result in results} == {"worker-0", "worker-1"}
    assert {result["value"] for result in results} == {1, 2}


def test_retinexformer_utils_bootstrap_breaks_scandir_circular_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "basicsr_retinexformer"
    utils_root = package_root / "utils"
    utils_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (utils_root / "misc.py").write_text(
        "def scandir(*args, **kwargs): return iter(())\n" "def scandir_SIDD(*args, **kwargs): return iter(())\n",
        encoding="utf-8",
    )
    (utils_root / "create_lmdb.py").write_text(
        "from basicsr_retinexformer.utils import scandir\n" "def create_lmdb_for_gopro(): return scandir\n",
        encoding="utf-8",
    )
    (utils_root / "__init__.py").write_text(
        "from .create_lmdb import create_lmdb_for_gopro\n" "from .misc import scandir, scandir_SIDD\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in list(sys.modules):
        if module_name == "basicsr_retinexformer" or module_name.startswith("basicsr_retinexformer."):
            monkeypatch.delitem(sys.modules, module_name)

    _bootstrap_retinexformer_utils(tmp_path)

    utils_module = sys.modules["basicsr_retinexformer.utils"]
    assert list(utils_module.scandir()) == []
    assert utils_module.create_lmdb_for_gopro() is utils_module.scandir


def test_generic_basicsr_utils_bootstrap_supports_namespace_package(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "basicsr"
    utils_root = package_root / "utils"
    utils_root.mkdir(parents=True)
    (utils_root / "misc.py").write_text(
        "def scandir(*args, **kwargs): return iter(())\n" "def scandir_SIDD(*args, **kwargs): return iter(())\n",
        encoding="utf-8",
    )
    (utils_root / "create_lmdb.py").write_text(
        "from basicsr.utils import scandir\n" "def create_lmdb_for_gopro(): return scandir\n",
        encoding="utf-8",
    )
    (utils_root / "__init__.py").write_text(
        "from .create_lmdb import create_lmdb_for_gopro\n" "from .misc import scandir, scandir_SIDD\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in list(sys.modules):
        if module_name == "basicsr" or module_name.startswith("basicsr."):
            monkeypatch.delitem(sys.modules, module_name)

    _bootstrap_basicsr_utils("basicsr", package_root)

    utils_module = sys.modules["basicsr.utils"]
    assert list(utils_module.scandir()) == []
    assert utils_module.create_lmdb_for_gopro() is utils_module.scandir


def test_persistent_service_http_protocol() -> None:
    class FakeRuntime:
        state = "ready"

        def restore(self, payload):
            return {"status": "success", "action": payload["action"]}

        def evaluate(self, payload):
            return {"status": "success", "image_path": payload["image_path"]}

        def health(self):
            return {"status": self.state}

        def sleep(self):
            self.state = "sleeping"
            return {"status": self.state}

        def wake(self):
            self.state = "ready"
            return {"status": self.state}

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(FakeRuntime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        restoration = post_json(base_url, "/restore", {"action": "scunet"}, 2)
        evaluation = post_json(base_url, "/evaluate", {"image_path": "/tmp/image.png"}, 2)
        sleeping = post_json(base_url, "/sleep", {}, 2)
        waking = post_json(base_url, "/wake", {}, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert restoration == {"status": "success", "action": "scunet"}
    assert evaluation == {"status": "success", "image_path": "/tmp/image.png"}
    assert sleeping == {"status": "sleeping"}
    assert waking == {"status": "ready"}


def test_subprocess_worker_publishes_only_valid_images(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import argparse, json, shutil\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--adapter')\n"
        "p.add_argument('--model'); p.add_argument('--external-tools-root'); p.add_argument('--device')\n"
        "a=p.parse_args(); shutil.copy2(a.input,a.output)\n"
        "print('RESULT_JSON='+json.dumps({'status':'success','model':a.model}))\n",
        encoding="utf-8",
    )
    settings = SubprocessSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
    )

    result = SubprocessRestorationWorker(settings, _registry()).restore(
        "test_tool", str(input_path), str(tmp_path / "output"), 0
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.output_path is not None
    assert Path(result.output_path).is_file()
    assert not list((tmp_path / "output").glob("*.partial.png"))


def test_subprocess_worker_rejects_and_cleans_corrupt_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--adapter')\n"
        "p.add_argument('--model'); p.add_argument('--external-tools-root'); p.add_argument('--device')\n"
        "a=p.parse_args(); open(a.output,'wb').write(b'broken')\n"
        "print('RESULT_JSON='+json.dumps({'status':'success'}))\n",
        encoding="utf-8",
    )
    settings = SubprocessSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
    )

    result = SubprocessRestorationWorker(settings, _registry()).restore(
        "test_tool", str(input_path), str(tmp_path / "output"), 0
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.output_path is None
    assert not list((tmp_path / "output").glob("*.partial.png"))


def test_restoration_worker_uses_persistent_service(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    entrypoint = tmp_path / "worker.py"
    _write_image(input_path)
    entrypoint.touch()

    def fake_post_json(base_url: str, endpoint: str, payload: dict[str, object], timeout: float):
        assert base_url == "http://127.0.0.1:8767"
        assert endpoint == "/restore"
        assert payload["action"] == "test_tool"
        assert timeout == 5
        Path(str(payload["output_path"])).write_bytes(input_path.read_bytes())
        return {"status": "success", "model": "test_model", "persistent": True}

    monkeypatch.setattr(worker_module, "post_json", fake_post_json)
    settings = SubprocessSettings(
        python_executable=sys.executable,
        service_url="http://127.0.0.1:8767",
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        device="cuda:1",
        timeout_seconds=5,
    )

    result = SubprocessRestorationWorker(settings, _registry()).restore(
        "test_tool", str(input_path), str(tmp_path / "output"), 0
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.metadata["persistent"] is True
    assert result.metadata["device"] == "cuda:1"
    assert result.metadata["service_url"] == "http://127.0.0.1:8767"


def test_pyiqa_evaluator_normalizes_and_aggregates_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "iqa.py"
    entrypoint.write_text(
        "import json\n"
        "print('RESULT_JSON='+json.dumps({'status':'success','raw_scores':"
        "{'topiq_nr':0.8,'musiq':60.0,'niqe':2.0}}))\n",
        encoding="utf-8",
    )
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
        metrics=[
            IQAMetricConfig(name="topiq_nr", weight=0.45, minimum=0, maximum=1),
            IQAMetricConfig(name="musiq", weight=0.35, minimum=0, maximum=100),
            IQAMetricConfig(name="niqe", weight=0.20, minimum=0, maximum=10, higher_is_better=False),
        ],
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.6, original_score=0.5, best_score=0.6
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.normalized_scores == {"topiq_nr": 0.8, "musiq": 0.6, "niqe": 0.8}
    assert abs(result.aggregate_score - 0.73) < 1e-9
    assert result.is_new_best is True


def test_pyiqa_evaluator_uses_persistent_service(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    entrypoint = tmp_path / "iqa.py"
    _write_image(input_path)
    entrypoint.touch()

    def fake_post_json(base_url: str, endpoint: str, payload: dict[str, object], timeout: float):
        assert base_url == "http://127.0.0.1:8767"
        assert endpoint == "/evaluate"
        assert payload["image_path"] == str(input_path.resolve())
        assert timeout == 5
        return {"status": "success", "raw_scores": {"maniqa": 0.75}, "persistent": True}

    monkeypatch.setattr(evaluator_module, "post_json", fake_post_json)
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        service_url="http://127.0.0.1:8767",
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        device="cuda:0",
        timeout_seconds=5,
        metrics=[IQAMetricConfig(name="maniqa", weight=1, minimum=0, maximum=1)],
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.5, original_score=0.5, best_score=0.5
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.aggregate_score == 0.75
    assert result.metadata["persistent"] is True
    assert result.metadata["device"] == "cuda:0"
    assert result.metadata["service_url"] == "http://127.0.0.1:8767"


def test_pyiqa_evaluator_failure_has_zero_gain(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        entrypoint=str(tmp_path / "missing.py"),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        metrics=[IQAMetricConfig(name="topiq_nr", weight=1, minimum=0, maximum=1)],
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.6, original_score=0.5, best_score=0.7
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.aggregate_score == 0.6
    assert result.delta_from_previous == 0.0
    assert result.is_new_best is False


def test_pyiqa_evaluator_applies_calibrated_direction_and_zscore(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "iqa.py"
    entrypoint.write_text(
        "import json\n"
        "print('RESULT_JSON='+json.dumps({'status':'success','raw_scores':"
        "{'maniqa':0.6,'niqe':2.0}}))\n",
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"metrics":{'
        '"maniqa":{"raw_transform":"identity","mean":0.5,"std":0.1,"weight":0.25},'
        '"niqe":{"raw_transform":"negate","mean":-4.0,"std":2.0,"weight":0.75}'
        "}}",
        encoding="utf-8",
    )
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
        reward_calibration_path=str(calibration),
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.0, original_score=0.0, best_score=0.0
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert abs(result.normalized_scores["maniqa"] - 1.0) < 1e-9
    assert abs(result.normalized_scores["niqe"] - 1.0) < 1e-9
    assert abs(result.aggregate_score - 1.0) < 1e-9
    assert result.metadata["normalization_mode"] == "zscore"
