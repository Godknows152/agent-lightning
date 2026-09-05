#!/usr/bin/env python3
"""Closed-loop 30-minute KL/entropy controller for ALFWorld Qwen3.5-2B."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import psutil

ROOT = Path("/home/LXJ/Python_Projects/Agent_Lightning")
BASE = ROOT / "examples/alfworld_grpo_baseline"
CONFIG = BASE / "config/alfworld/qwen35_2b/v1/alfworld_config_2gpu.yaml"
OUTPUT_ROOT = BASE / "outputs/alfworld/qwen35_2b/v1/2gpu"
LOG_ROOT = BASE / "log/alfworld/qwen35_2b/v1/2gpu"
ADAPTIVE_ROOT = OUTPUT_ROOT / "adaptive"
DOC = BASE / "ADAPTIVE_KL_ENTROPY_QWEN35_2B.md"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "agent-lightning"
STATE_FILE = STATE_DIR / "alfworld_qwen35_2b_adaptive.json"
MONITOR_LOG = STATE_DIR / "alfworld_qwen35_2b_adaptive.log"
LAUNCHER = BASE / "scripts/alfworld/qwen35_2b_v1.sh"
PROJECT = "Godknows/ALFWorldRL"
RUN_NAME = "alfworld_qwen35_2b_v1_seed0"
SWANLAB_BIN = Path(os.environ.get("SWANLAB_BIN", "/home/LXJ/anaconda3/envs/alfworld-verl/bin/swanlab"))
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", "/home/LXJ/anaconda3/envs/alfworld-verl/bin/python"))

TRAINING_UNIT = "agent-lightning-alfworld-qwen35-2b-training.service"

WINDOW = 3
STARTUP_GRACE = 12 * 60
RESTART_COOLDOWN = 10 * 60
NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
KEYS = ("actor/entropy", "critic/rewards/mean", "actor/tool_choice_entropy", "actor/action_path_entropy", "actor/kl_loss")


def log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"attempt": 0, "current_output_dir": None, "cloud_run_id": None, "last_restart_at": None}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"attempt": 0, "current_output_dir": None, "cloud_run_id": None, "last_restart_at": None}


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def coefficient(key: str) -> float:
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*({NUMBER})(?:\s+#.*)?\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {key} in {CONFIG}")
    return float(match.group(1))


def params() -> dict[str, float]:
    return {"kl_loss_coef": coefficient("kl_loss_coef"), "entropy_coeff": coefficient("entropy_coeff")}


def set_coefficient(text: str, key: str, value: float) -> str:
    pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*{NUMBER}(\s+#.*)?\s*$", re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(f"cannot find {key} for update")
    return pattern.sub(lambda m: f"{m.group(1)}{key}: {value:.12g}{m.group(2) or ''}", text, count=1)


def update_config(new: dict[str, float]) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    text = set_coefficient(text, "kl_loss_coef", new["kl_loss_coef"])
    text = set_coefficient(text, "entropy_coeff", new["entropy_coeff"])
    CONFIG.write_text(text, encoding="utf-8")


def json_from_output(raw: str) -> dict[str, Any]:
    index = raw.find("{")
    if index < 0:
        raise ValueError("no JSON object returned")
    return json.JSONDecoder().raw_decode(raw[index:])[0]


def cloud_runs() -> list[dict[str, Any]]:
    raw = subprocess.check_output([str(SWANLAB_BIN), "api", "run", "list", PROJECT, "-n", "1", "-s", "100"], text=True, stderr=subprocess.DEVNULL)
    return json_from_output(raw).get("data", {}).get("list", [])


def cloud_run_id(state: dict[str, Any]) -> str | None:
    if state.get("cloud_run_id"):
        return str(state["cloud_run_id"])
    started = float(state.get("last_restart_at") or 0)
    choices: list[tuple[float, str]] = []
    for run in cloud_runs():
        if run.get("name") != RUN_NAME or not run.get("run_id"):
            continue
        try:
            created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if not started or created >= started - 15 * 60:
            choices.append((abs(created - started), str(run["run_id"])))
    return min(choices)[1] if choices else None


def cloud_metrics(run_id: str | None) -> list[dict[str, float]]:
    if not run_id:
        return []
    try:
        raw = subprocess.check_output(
            [str(SWANLAB_BIN), "api", "run", "metrics", f"{PROJECT}/{run_id}", "--keys", ",".join(KEYS), "--range-tail", "20"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json_from_output(raw).get("data", {}).get("list", [])
    except Exception as exc:
        log(f"SwanLab metrics unavailable: {exc}")
        return []
    by_step: dict[int, dict[str, float]] = {}
    for item in payload:
        key = item.get("key")
        for point in item.get("metrics", []):
            if key in KEYS and point.get("step") is not None:
                by_step.setdefault(int(point["step"]), {})[key] = float(point["value"])
    rows = []
    for step in sorted(by_step):
        if "actor/entropy" in by_step[step] and "critic/rewards/mean" in by_step[step]:
            row = dict(by_step[step])
            row["step"] = float(step)
            rows.append(row)
    return rows


def local_metrics(output_dir: Path | None) -> list[dict[str, float]]:
    if not output_dir:
        return []
    rows: list[dict[str, float]] = []
    pattern = re.compile(r"(?P<key>actor/entropy|critic/rewards/mean|actor/tool_choice_entropy|actor/action_path_entropy|actor/kl_loss|training/global_step):(?P<value>" + NUMBER + r")")
    for path in sorted(output_dir.rglob("*.log"), key=lambda p: p.stat().st_mtime):
        for line in path.open(encoding="utf-8", errors="ignore"):
            if "actor/entropy:" not in line:
                continue
            row = {m.group("key"): float(m.group("value")) for m in pattern.finditer(line)}
            if "actor/entropy" in row and "critic/rewards/mean" in row:
                row["step"] = row.get("training/global_step", float(len(rows) + 1))
                rows.append(row)
    return rows


def training_status() -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", TRAINING_UNIT,
         "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "MainPID"],
        text=True, capture_output=True, check=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def matching_processes() -> list[psutil.Process]:
    # Do not match shell command strings or other ALFWorld model profiles.
    result = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        args = proc.info["cmdline"] or []
        if "alfworld_baseline.main_ppo" in args and str(CONFIG.parent) in args:
            result.append(proc)
        elif str(LAUNCHER) in args and proc.pid != os.getpid():
            result.append(proc)
    return result


def stop_training() -> None:
    # systemd owns the complete Ray/SGLang cgroup, including reparented workers.
    subprocess.run(["systemctl", "--user", "stop", TRAINING_UNIT], check=True, timeout=150)
    if training_status().get("ActiveState") in {"active", "activating", "deactivating"}:
        raise RuntimeError("training unit did not stop; refusing cleanup")
    if matching_processes():
        raise RuntimeError("unmanaged Qwen3.5-2B training remains; refusing cleanup")


def delete_cloud(run_id: str | None) -> None:
    if not run_id:
        return
    code = f"from swanlab import Api; r=Api().run('{PROJECT}/{run_id}'); print(r.delete(commit=True))"
    result = subprocess.run([str(PYTHON_BIN), "-c", code], text=True, capture_output=True)
    if result.returncode != 0 or "True" not in result.stdout:
        raise RuntimeError(f"SwanLab deletion failed for {run_id}: {result.stdout} {result.stderr}")
    log(f"deleted SwanLab run {run_id}")


def cleanup(output_dir: Path | None, run_id: str | None) -> None:
    delete_cloud(run_id)
    if output_dir and output_dir.exists():
        resolved = output_dir.resolve()
        if OUTPUT_ROOT.resolve() not in resolved.parents or resolved == OUTPUT_ROOT.resolve():
            raise RuntimeError(f"unsafe ALFWorld output path: {resolved}")
        shutil.rmtree(resolved)
        log(f"deleted failed ALFWorld output {resolved}")


def adjust(old: dict[str, float], reason: str) -> dict[str, float]:
    if reason == "entropy_collapse":
        entropy, kl = old["entropy_coeff"] * 1.5, old["kl_loss_coef"] * 0.75
    elif reason == "entropy_explosion":
        entropy, kl = old["entropy_coeff"] * 0.5, old["kl_loss_coef"] * 1.5
    else:
        entropy, kl = old["entropy_coeff"] * 1.15, old["kl_loss_coef"] * 0.80
    return {"entropy_coeff": min(0.05, max(0.0005, entropy)), "kl_loss_coef": min(0.02, max(0.0001, kl))}


def append_doc(old: dict[str, float], new: dict[str, float], reason: str, metrics: dict[str, Any]) -> None:
    if not DOC.exists():
        DOC.write_text(
            "# ALFWorld Qwen3.5-2B 自适应 KL/Entropy 记录\n\n"
            "每 30 分钟检查一次，只修改 `actor.kl_loss_coef` 与 `actor.entropy_coeff`。熵坍缩阈值：连续窗口低于 0.1 或工具探索归零；熵爆炸阈值：连续窗口高于 12；奖励要求窗口中位数持续改善。\n\n"
            "| 时间 | 原因 | 调整前（KL / Entropy） | 调整后（KL / Entropy） | 上一轮效果 |\n|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    effect = f"step={metrics.get('step')}, reward={metrics.get('reward')}, entropy={metrics.get('entropy')}, tool_entropy={metrics.get('tool_entropy')}"
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    with DOC.open("a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | {reason} | {old['kl_loss_coef']:.6g} / {old['entropy_coeff']:.6g} | {new['kl_loss_coef']:.6g} / {new['entropy_coeff']:.6g} | {effect} |\n")


def commit(reason: str) -> None:
    paths = [str(CONFIG.relative_to(ROOT)), str(DOC.relative_to(ROOT))]
    result = subprocess.run(["git", "commit", "--only", "-m", f"chore(alfworld): adapt KL and entropy ({reason})", "--", *paths], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git commit failed: {result.stdout}\n{result.stderr}")
    log(f"committed {reason}")


def launch(state: dict[str, Any], new: dict[str, float], *, retry: bool = False) -> None:
    if retry:
        output = Path(state["current_output_dir"])
    else:
        attempt = int(state.get("attempt", 0)) + 1
        name = f"attempt_{attempt:03d}_kl{new['kl_loss_coef']:.5f}_e{new['entropy_coeff']:.5f}"
        output = ADAPTIVE_ROOT / name
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing attempt: {output}")
    if training_status().get("ActiveState") in {"active", "activating", "deactivating"}:
        raise RuntimeError("training already active")
    if matching_processes():
        raise RuntimeError("unmanaged target training already active")
    output.mkdir(parents=True, exist_ok=True)
    (output / "log").mkdir(exist_ok=True)
    torch_lib = PYTHON_BIN.parent.parent / "lib/python3.12/site-packages/torch/lib"
    cuda_runtime_lib = PYTHON_BIN.parent.parent / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env = {
        "ALFWORLD_MODEL_PROFILE": "qwen35_2b",
        "ALFWORLD_OUTPUT_DIR": str(output),
        "ALFWORLD_LOG_DIR": str(output / "log"),
        "ALFWORLD_SWANLAB_LOG_DIR": str(output / "swanlab"),
        "ALFWORLD_SWANLAB_MODE": "cloud",
        "ALFWORLD_FOREGROUND": "1",
        "ALFWORLD_TOTAL_STEPS": "150",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHON_BIN": str(PYTHON_BIN),
        "SWANLAB_BIN": str(SWANLAB_BIN),
        # Keep PyTorch's bundled CUDA runtime ahead of host CUDA.  The SGLang
        # spawn path otherwise may resolve an incompatible libcudart and die
        # before the first training step.
        "LD_LIBRARY_PATH": f"{torch_lib}:{cuda_runtime_lib}:/usr/local/cuda/lib64",
    }
    command = [
        "systemd-run", "--user", f"--unit={TRAINING_UNIT}",
        "--service-type=exec", f"--working-directory={ROOT}",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=90",
        f"--property=StandardOutput=append:{output / 'log' / 'training.log'}",
        f"--property=StandardError=append:{output / 'log' / 'training.log'}",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    command.extend(["/bin/bash", str(LAUNCHER)])
    subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
    if not retry:
        state["attempt"] = attempt
    state.update({"current_output_dir": str(output), "cloud_run_id": None,
                  "last_restart_at": time.time(), "params": new, "training_unit": TRAINING_UNIT})
    save_state(state)
    log(f"training service dispatched for attempt {state['attempt']}; waiting for actual metrics")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-start", action="store_true", help="retry an empty launch without changing parameters")
    args = parser.parse_args()
    state = load_state()
    old = params()
    output = Path(state["current_output_dir"]) if state.get("current_output_dir") else None
    if args.retry_start:
        if output is None or not output.is_relative_to(ADAPTIVE_ROOT):
            raise RuntimeError("no managed attempt to retry")
        # Only an attempt that never reached training may reuse its directory.
        if any(p.stat().st_size for p in output.rglob("*") if p.is_file()):
            raise RuntimeError("attempt contains data; inspect before retrying")
        if not args.dry_run:
            launch(state, old, retry=True)
        return 0
    status = training_status()
    alive = status.get("ActiveState") in {"active", "activating"} or bool(matching_processes())
    rows = local_metrics(output)
    run_id = state.get("cloud_run_id")
    if not rows and (alive or output):
        run_id = cloud_run_id(state)
        rows = cloud_metrics(run_id)
    reason: str | None = None
    metrics: dict[str, Any] = {"rows": len(rows)}
    if len(rows) >= WINDOW:
        recent = rows[-WINDOW:]
        prior = rows[-2 * WINDOW:-WINDOW] if len(rows) >= 2 * WINDOW else []
        ent = [r["actor/entropy"] for r in recent]
        tool = [r.get("actor/tool_choice_entropy", 0.0) for r in recent]
        metrics.update({"step": int(recent[-1]["step"]), "reward": round(median(r["critic/rewards/mean"] for r in recent), 6), "entropy": round(median(ent), 6), "tool_entropy": round(median(tool), 6)})
        if sum(v < 0.1 for v in ent) >= 2 or all(v <= 0 for v in tool):
            reason = "entropy_collapse"
        elif sum(v > 12 for v in ent) >= 2:
            reason = "entropy_explosion"
        elif len(prior) == WINDOW and median(r["critic/rewards/mean"] for r in recent) <= median(r["critic/rewards/mean"] for r in prior):
            reason = "reward_not_rising"
    if reason is None and not alive and state.get("last_restart_at"):
        age = time.time() - float(state["last_restart_at"])
        if age >= STARTUP_GRACE:
            reason = "failed_or_no_metrics"
    if reason is None and not alive and not state.get("last_restart_at"):
        reason = "bootstrap"
    log(f"check alive={alive} rows={len(rows)} reason={reason or 'hold'} metrics={metrics} params={old}")
    if reason in {"failed_or_no_metrics", "bootstrap"}:
        # Infrastructure failure is not evidence that KL/entropy is wrong.
        log("attention: no running training or insufficient metrics; parameters unchanged")
        return 0
    if not reason or args.dry_run:
        if args.dry_run and reason:
            log(f"dry-run would apply {adjust(old, reason)}")
        return 0
    if state.get("last_restart_at") and time.time() - float(state["last_restart_at"]) < RESTART_COOLDOWN:
        log("restart suppressed by cooldown")
        return 0
    stop_training()
    cleanup(output, run_id or cloud_run_id(state))
    new = adjust(old, reason)
    update_config(new)
    append_doc(old, new, reason, metrics)
    commit(reason)
    launch(state, new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
