#!/usr/bin/env python3
"""30-minute adaptive controller for the current fog v4.1.4 GRPO run.

The controller deliberately changes only KL/entropy coefficients.  It reads the
VERL text log, keeps a small state file outside git, removes the failed attempt
(and its SwanLab run), commits the coefficient change, and starts a fresh run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - the host environment has psutil
    psutil = None

ROOT = Path("/home/LXJ/Python_Projects/Agent_Lightning")
OLD_VERL = ROOT / "examples/image_restoration_multi_agent/old_verl_grpo"
CONFIG = OLD_VERL / "config/fog/v4.1.4/fog_config_2gpu.yaml"
LOG_DIR = OLD_VERL / "log/fog/v4.1.4/2gpu"
OUTPUT_ROOT = OLD_VERL / "outputs/fog/v4.1.4/2gpu"
ADAPTIVE_ROOT = OUTPUT_ROOT / "adaptive"
DOC = OLD_VERL / "ADAPTIVE_TRAINING_PARAMETER_UPDATES.md"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "agent-lightning"
STATE = STATE_DIR / "fog_v414_adaptive_monitor.json"
MONITOR_LOG = STATE_DIR / "fog_v414_adaptive_monitor.log"
RUN_SCRIPT = OLD_VERL / "scripts/fog/fog_v4_1_4.sh"
PROJECT_PATH = "Godknows/FogRL"
EXPERIMENT_NAME = "fog_v4.1.4"

WINDOW = 3
ENTROPY_LOW = 0.35
ENTROPY_HIGH = 1.20
ACTION_ENTROPY_LOW = 0.08
ACTION_ENTROPY_HIGH = 0.70
MIN_REWARD_DELTA = 0.02
MIN_RESTART_GAP_SECONDS = 10 * 60
STARTUP_GRACE_SECONDS = 12 * 60

NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(
    rf"(?P<key>actor/entropy|critic/rewards/mean|"
    rf"actor/decision_point_first_token_action_entropy_normalized|"
    rf"actor/kl_loss|actor/kl_coef|training/global_step):(?P<value>{NUMBER})"
)
STEP_RE = re.compile(r"\bstep:(\d+)\b")
START_RE = re.compile(r"run-(\d{8}_\d{6})-")


def log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_state() -> dict[str, Any]:
    if not STATE.exists():
        return {"attempt": 0, "current_output_dir": None, "current_run_id": None, "last_restart_at": None}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"state unreadable, resetting: {exc}")
        return {"attempt": 0, "current_output_dir": None, "current_run_id": None, "last_restart_at": None}


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def parse_coeff(text: str, key: str) -> float | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*({NUMBER})(?:\s+#.*)?\s*$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def effective_params() -> dict[str, float]:
    common = (OLD_VERL / "config/restoration_common_config_2gpu.yaml").read_text(encoding="utf-8")
    fog = CONFIG.read_text(encoding="utf-8")
    # Per-expert overrides in fog take precedence; otherwise inherit common.
    kl = parse_coeff(fog, "kl_coef") or parse_coeff(common, "kl_coef")
    kl_loss = parse_coeff(fog, "kl_loss_coef") or parse_coeff(common, "kl_loss_coef")
    entropy = parse_coeff(fog, "entropy_coeff")
    first = parse_coeff(fog, "decision_point_first_token_entropy_coeff")
    if None in (kl, kl_loss, entropy, first):
        raise RuntimeError(f"could not read current KL/entropy coefficients from {CONFIG}")
    return {"kl_coef": kl, "kl_loss_coef": kl_loss, "entropy_coeff": entropy, "first_token_entropy_coeff": first}


def replace_or_insert(text: str, key: str, value: float, section: str) -> str:
    rendered = f"{value:.12g}"
    pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*{NUMBER}\s*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{key}: {rendered}", text, count=1)
    anchor = re.compile(rf"^(\s*){re.escape(section)}:\s*.*$", re.MULTILINE)
    match = anchor.search(text)
    if not match:
        raise RuntimeError(f"section {section!r} not found while adding {key}")
    indent = match.group(1) + "  "
    pos = match.end()
    return text[:pos] + f"\n{indent}{key}: {rendered}" + text[pos:]


def update_config(params: dict[str, float]) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    # All edits are coefficient-only and are scoped to the fog config.  This
    # prevents adaptive fog runs from silently changing other experts.
    text = replace_or_insert(text, "kl_loss_coef", params["kl_loss_coef"], "actor")
    text = replace_or_insert(text, "entropy_coeff", params["entropy_coeff"], "actor")
    # The first-token key already exists in the current v4.1.4 config.
    first_pattern = re.compile(r"^(\s*)decision_point_first_token_entropy_coeff:\s*" + NUMBER + r"\s*$", re.MULTILINE)
    if first_pattern.search(text):
        text = first_pattern.sub(lambda m: f"{m.group(1)}decision_point_first_token_entropy_coeff: {params['first_token_entropy_coeff']:.12g}", text, count=1)
    else:
        raise RuntimeError("decision_point_first_token_entropy_coeff is missing from fog config")
    # Add a per-expert algorithm override if it is not present.  This leaves
    # restoration_common_config untouched for the other experts.
    if not re.search(r"^algorithm:\s*$", text, re.MULTILINE):
        marker = "\ntrainer:\n"
        if marker not in text:
            raise RuntimeError("trainer section not found while adding algorithm override")
        block = (
            "\nalgorithm:\n"
            "  kl_ctrl:\n"
            f"    kl_coef: {params['kl_coef']:.12g}\n"
        )
        text = text.replace(marker, block + marker, 1)
    else:
        kl_pattern = re.compile(r"^(\s{4})kl_coef:\s*" + NUMBER + r"\s*$", re.MULTILINE)
        if kl_pattern.search(text):
            text = kl_pattern.sub(lambda m: f"{m.group(1)}kl_coef: {params['kl_coef']:.12g}", text, count=1)
        else:
            text = text.replace("algorithm:\n", "algorithm:\n  kl_ctrl:\n    kl_coef: " + f"{params['kl_coef']:.12g}\n", 1)
    CONFIG.write_text(text, encoding="utf-8")


def append_doc(previous: dict[str, float], current: dict[str, float], reason: str, metrics: dict[str, Any]) -> None:
    if not DOC.exists():
        DOC.write_text(
            "# 自适应训练参数更新记录\n\n"
            "监控对象：fog v4.1.4（2 GPU）。每 30 分钟检查一次；仅允许调整 KL 与 Entropy 系数。\n\n"
            "判据：`actor/entropy` 安全区间为 [0.35, 1.20]，首 Token 动作熵归一化值安全区间为 [0.08, 0.70]；奖励使用最近 3 步与前 3 步的中位数比较。\n\n"
            "| 时间 | 原因 | 修改前（KL / KL-loss / Entropy / 首Token Entropy） | 修改后 | 上一轮效果 |\n"
            "|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    effect = (
        f"step={metrics.get('step')}, reward={metrics.get('reward')}, "
        f"entropy={metrics.get('entropy')}, action_entropy={metrics.get('action_entropy')}"
    )
    before = ", ".join(f"{previous[k]:.6g}" for k in ("kl_coef", "kl_loss_coef", "entropy_coeff", "first_token_entropy_coeff"))
    after = ", ".join(f"{current[k]:.6g}" for k in ("kl_coef", "kl_loss_coef", "entropy_coeff", "first_token_entropy_coeff"))
    with DOC.open("a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | {reason} | {before} | {after} | {effect} |\n")


def git_commit(reason: str) -> None:
    message = f"chore(restoration): adapt KL and entropy ({reason})"
    paths = [str(CONFIG.relative_to(ROOT)), str(DOC.relative_to(ROOT))]
    result = subprocess.run(["git", "commit", "--only", "-m", message, "--", *paths], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"git commit failed: {result.stdout}\n{result.stderr}")
    log(f"committed coefficient update: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else message}")


def metric_rows() -> list[dict[str, float]]:
    logs = sorted(LOG_DIR.glob("fog_v4.1.4_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return []
    path = logs[0]
    rows: list[dict[str, float]] = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        if "actor/entropy:" not in line or "training/global_step:" not in line:
            continue
        row = {m.group("key"): float(m.group("value")) for m in METRIC_RE.finditer(line)}
        if len(row) >= 4:
            row["step"] = row.get("training/global_step", float(len(rows) + 1))
            rows.append(row)
    return rows


def process_matches() -> list[Any]:
    if psutil is None:
        result = subprocess.run(["pgrep", "-af", "fog_v4_1_4|fog_config_2gpu|run_expert_old_verl_grpo_2gpu.sh fog"], text=True, capture_output=True)
        return result.stdout.splitlines() if result.returncode == 0 else []
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid == os.getpid():
            continue
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(token in cmd for token in ("fog_v4_1_4", "fog_config_2gpu", "run_expert_old_verl_grpo_2gpu.sh fog")):
            found.append(proc)
    return found


def stop_training() -> None:
    procs = process_matches()
    if not procs:
        return
    log(f"stopping {len(procs)} fog training processes before restart")
    for proc in procs:
        try:
            proc.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    deadline = time.time() + 90
    while time.time() < deadline:
        alive = [p for p in procs if p.is_running()]
        if not alive:
            return
        time.sleep(2)
    for proc in procs:
        try:
            if proc.is_running():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def local_run_id(output_dir: Path | None) -> str | None:
    if not output_dir:
        return None
    info = output_dir / ".swanlab_experiment.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text(encoding="utf-8")).get("run_id")
    except Exception:
        return None


def cloud_run_for_output(output_dir: Path | None) -> str | None:
    if not output_dir:
        return None
    candidates = list((output_dir / "swanlab").glob("run-*/files/swanlab-metadata.json"))
    if not candidates:
        candidates = list((OUTPUT_ROOT / "swanlab").glob("run-*/files/swanlab-metadata.json"))
    starts: list[datetime] = []
    for meta in candidates:
        match = START_RE.search(str(meta.parent.parent.name))
        if match:
            starts.append(datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc))
    if not starts:
        return None
    start = max(starts)
    try:
        raw = subprocess.check_output([str(Path(sys.executable).parent / "swanlab"), "api", "run", "list", PROJECT_PATH, "-n", "1", "-s", "100"], text=True, stderr=subprocess.DEVNULL)
        # The global SwanLab CLI may emit a Python warning before its JSON.
        # Decode from the first JSON object so cleanup remains reliable.
        start_index = raw.find("{")
        if start_index < 0:
            raise ValueError("SwanLab list returned no JSON object")
        runs = json.JSONDecoder().raw_decode(raw[start_index:])[0].get("data", {}).get("list", [])
    except Exception as exc:
        log(f"could not query SwanLab for deletion: {exc}")
        return None
    matching = []
    for run in runs:
        if run.get("name") != EXPERIMENT_NAME or not run.get("run_id"):
            continue
        try:
            created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        matching.append((abs((created - start).total_seconds()), run["run_id"], created))
    if not matching:
        return None
    distance, run_id, created = min(matching)
    return run_id if distance <= 15 * 60 else None


def delete_cloud_run(run_id: str | None) -> None:
    if not run_id:
        log("no matching SwanLab cloud run found; local cleanup will continue")
        return
    code = (
        "from swanlab import Api; "
        f"r=Api().run('{PROJECT_PATH}/{run_id}'); "
        "ok=r.delete(commit=True); print('deleted' if ok else 'delete_failed')"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if result.returncode != 0 or "deleted" not in result.stdout:
        raise RuntimeError(f"SwanLab deletion failed for {run_id}: {result.stdout} {result.stderr}")
    log(f"deleted SwanLab run {run_id}")


def cleanup_failed_attempt(output_dir: Path | None, cloud_id: str | None, local_id: str | None) -> None:
    delete_cloud_run(cloud_id)
    if output_dir and output_dir.exists():
        resolved = output_dir.resolve()
        if OUTPUT_ROOT.resolve() not in resolved.parents or resolved == OUTPUT_ROOT.resolve():
            raise RuntimeError(f"refusing unsafe output deletion: {resolved}")
        shutil.rmtree(resolved)
        log(f"deleted failed local output {resolved}")
    # Local SwanLab stores runs in a shared sibling directory.  The local run
    # id is present in the failed attempt's .swanlab_experiment.json, so remove
    # exactly that directory and retain all sibling attempts.
    if local_id:
        for run_dir in (OUTPUT_ROOT / "swanlab").glob(f"run-*{local_id}"):
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
                log(f"deleted failed local SwanLab output {run_dir}")


def latest_output_dir() -> Path | None:
    dirs = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name != "adaptive" and (p / ".swanlab_experiment.json").exists()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def decision(rows: list[dict[str, float]], process_alive: bool) -> tuple[str | None, dict[str, Any]]:
    if len(rows) < WINDOW:
        return ("failed_or_no_metrics" if not process_alive and rows else None), {"rows": len(rows)}
    recent = rows[-WINDOW:]
    prior = rows[-2 * WINDOW : -WINDOW] if len(rows) >= 2 * WINDOW else rows[:WINDOW]
    ent = [r["actor/entropy"] for r in recent]
    act = [r["actor/decision_point_first_token_action_entropy_normalized"] for r in recent if "actor/decision_point_first_token_action_entropy_normalized" in r]
    reward_recent = median([r["critic/rewards/mean"] for r in recent])
    reward_prior = median([r["critic/rewards/mean"] for r in prior])
    metrics = {
        "step": int(recent[-1]["step"]),
        "reward": round(reward_recent, 6),
        "reward_prior": round(reward_prior, 6),
        "entropy": round(median(ent), 6),
        "action_entropy": round(median(act), 6) if act else None,
        "rows": len(rows),
    }
    if median(ent) < ENTROPY_LOW or (act and median(act) < ACTION_ENTROPY_LOW):
        return "entropy_collapse", metrics
    if median(ent) > ENTROPY_HIGH or (act and median(act) > ACTION_ENTROPY_HIGH):
        return "entropy_explosion", metrics
    if len(rows) >= 2 * WINDOW and reward_recent <= reward_prior + MIN_REWARD_DELTA:
        return "reward_not_rising", metrics
    if not process_alive:
        return "failed_or_finished", metrics
    return None, metrics


def adjusted_params(old: dict[str, float], reason: str) -> dict[str, float]:
    if reason == "entropy_collapse":
        kl_factor, ent_factor, first_factor = 1.25, 1.50, 1.40
    elif reason == "entropy_explosion":
        kl_factor, ent_factor, first_factor = 1.25, 0.50, 0.60
    else:  # stagnant, failed, or finished: loosen KL and add measured exploration.
        kl_factor, ent_factor, first_factor = 0.75, 1.20, 1.15
    kl = min(0.20, max(0.01, old["kl_coef"] * kl_factor))
    entropy = min(0.02, max(0.0001, old["entropy_coeff"] * ent_factor))
    first = min(0.03, max(0.0005, old["first_token_entropy_coeff"] * first_factor))
    return {"kl_coef": kl, "kl_loss_coef": kl, "entropy_coeff": entropy, "first_token_entropy_coeff": first}


def launch(state: dict[str, Any], params: dict[str, float]) -> Path:
    state["attempt"] = int(state.get("attempt", 0)) + 1
    name = f"attempt_{state['attempt']:03d}_kl{params['kl_coef']:.5f}_e{params['entropy_coeff']:.5f}_first{params['first_token_entropy_coeff']:.5f}"
    output = ADAPTIVE_ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OLD_VERL_OUTPUT_DIR": str(output),
            "OLD_VERL_SWANLAB_LOG_DIR": str(output / "swanlab"),
            "OLD_VERL_RESUME_MODE": "disable",
            "OLD_VERL_RESUME_FROM_PATH": "",
            "OLD_VERL_RUN_IN_FOREGROUND": "0",
        }
    )
    subprocess.Popen(["bash", str(RUN_SCRIPT)], cwd=ROOT, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state.update({"current_output_dir": str(output), "current_run_id": None, "last_restart_at": time.time(), "params": params})
    save_state(state)
    log(f"started fresh run in {output} with {params}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one check; default is also one check")
    parser.add_argument("--dry-run", action="store_true", help="report decision without changing files or starting training")
    args = parser.parse_args()
    state = load_state()
    params = effective_params()
    current_output = Path(state["current_output_dir"]) if state.get("current_output_dir") else latest_output_dir()
    process_alive = bool(process_matches())
    rows = metric_rows()
    reason, metrics = decision(rows, process_alive)
    if (
        reason is None
        and not process_alive
        and state.get("current_output_dir")
        and state.get("last_restart_at")
        and time.time() - float(state["last_restart_at"]) >= STARTUP_GRACE_SECONDS
    ):
        reason, metrics = "failed_or_no_metrics", {"rows": len(rows)}
    log(f"check: alive={process_alive} rows={len(rows)} reason={reason or 'hold'} metrics={metrics} params={params}")
    if not reason:
        return 0
    if state.get("last_restart_at") and time.time() - float(state["last_restart_at"]) < MIN_RESTART_GAP_SECONDS:
        log("restart suppressed by cooldown")
        return 0
    if args.dry_run:
        log(f"dry-run would apply {adjusted_params(params, reason)}")
        return 0
    stop_training()
    local_id = local_run_id(current_output)
    cloud_id = state.get("current_cloud_run_id") or cloud_run_for_output(current_output)
    next_params = adjusted_params(params, reason)
    cleanup_failed_attempt(current_output, cloud_id, local_id)
    update_config(next_params)
    append_doc(params, next_params, reason, metrics)
    git_commit(reason)
    state.update({"current_cloud_run_id": None, "last_reason": reason, "last_effect": metrics})
    launch(state, next_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
