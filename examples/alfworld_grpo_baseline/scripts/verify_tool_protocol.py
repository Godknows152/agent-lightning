#!/usr/bin/env python3
"""Verify GiGPO context construction and the Qwen3 XML tool-call protocol.

This diagnostic is intentionally independent of Ray/VERL.  It renders the
exact chat template used by the isolated baseline and, when requested,
generates a few first-turn responses with the downloaded Qwen model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

# Qwen3.5 support is vendored by the old-VERL image-restoration setup.
OLD_VERL_PYDEPS = Path("/home/LXJ/Python_Projects/Agent_Lightning/examples/image_restoration_multi_agent/old_verl_grpo/.pydeps")
if OLD_VERL_PYDEPS.is_dir():
    sys.path.insert(0, str(OLD_VERL_PYDEPS))

import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alfworld_baseline.prompts_gigpo import (
    NONTHINKING_PROMPT_VERSION, PROMPT_VERSION, QWEN3_ALFWORLD_CHAT_TEMPLATE, build_user_prompt,
)
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from alfworld_baseline.xml_actions import parse_xml_decision

PROFILES = {
    "qwen25_1_5b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen2.5-1.5B-Instruct"),
        "data": ROOT / "data" / "qwen25_1_5b" / "train.parquet",
        "template": "GiGPO context / Qwen3 XML chat_template",
    },
    "qwen35_9b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen3.5-9B"),
        "data": ROOT / "data" / "qwen35_2b" / "train.parquet",
        "template": "GiGPO context / Qwen3 XML chat_template",
    },
    "qwen35_2b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen3.5-2B"),
        "data": ROOT / "data" / "qwen35_2b" / "train.parquet",
        "template": "GiGPO context / Qwen3 XML chat_template",
    },
}
RUNTIME_TERMINATION_MARKERS = ("<|im_end|>", "<|endoftext|>")
STRICT_XML_RE = re.compile(
    r"^<tool_call>\s*<function=alfworld_action>\s*"
    r"<parameter=action>\s*(.*?)\s*</parameter>\s*"
    r"</function>\s*</tool_call>$",
    re.DOTALL,
)


def strip_runtime_termination(text: str) -> tuple[str, list[str]]:
    """Remove only Qwen EOS/padding markers appended after a response.

    SGLang/Transformers may return the stop token in ``output_ids``.  These
    markers are transport-level delimiters, not model-visible text, so they
    must not turn an otherwise exact XML call into a format failure.  Any
    ordinary suffix is deliberately preserved and remains an error.
    """

    visible = text.rstrip()
    removed: list[str] = []
    while True:
        for marker in RUNTIME_TERMINATION_MARKERS:
            if visible.endswith(marker):
                removed.append(marker)
                visible = visible[: -len(marker)].rstrip()
                break
        else:
            return visible, removed


def classify(text: str) -> str:
    stripped, _ = strip_runtime_termination(text)
    if STRICT_XML_RE.fullmatch(stripped):
        return "qwen3_xml_strict"
    if "<tool_call>" in text and "</tool_call>" in text and "<function=" in text and "<parameter=" in text:
        return "qwen3_xml_complete_with_extra_text"
    if "<tool_call>" in text:
        return "qwen3_xml_incomplete"
    return "other"


def classify_visible_response(text: str) -> str:
    """Classify after transport EOS removal (public diagnostic helper)."""

    return classify(text)


def _admissible_actions(messages: list[dict[str, object]]) -> tuple[str, ...]:
    """Extract the action list injected by ``build_user_prompt``."""

    user = next((str(m.get("content", "")) for m in messages if m.get("role") == "user"), "")
    markers = (
        "Your admissible actions of the current situation are: [",
        "Current admissible actions (the action value must be copied exactly from this list):\n",
        "Current admissible actions (copy exactly one):\n",
    )
    marker = next((candidate for candidate in markers if candidate in user), None)
    if marker is None:
        return ()
    section = user.split(marker, 1)[1].split("\n\n", 1)[0]
    if marker.startswith("Your admissible"):
        return tuple(re.findall(r"'([^']*)'", section))
    return tuple(line[2:] for line in section.splitlines() if line.startswith("- "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="qwen35_2b")
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--avoid-repeated-actions", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    thinking = args.enable_thinking if args.enable_thinking is not None else True
    avoid_repeated = args.avoid_repeated_actions if args.avoid_repeated_actions is not None else args.profile == "qwen35_2b"

    profile = PROFILES[args.profile]
    model_path = profile["model"]
    data_path = profile["data"]
    output_path = ROOT / "outputs" / "diagnostics" / args.profile / "tool_protocol.json"

    row = pd.read_parquet(data_path).iloc[0]
    messages = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
    messages = [dict(x) for x in messages]
    admissible_actions = _admissible_actions(messages)
    if not admissible_actions:
        raise RuntimeError("could not extract admissible actions from the prepared prompt")
    tools = [ALFWorldToolRegistry.static_tool_schema()]
    # Saved parquet may contain an older prompt. Runtime rebuilds the first
    # request with the GiGPO profile and current admissible action list.
    user = next(str(m["content"]) for m in messages if m["role"] == "user")
    if "Task goal (not an executable action):" in user:
        mission = user.split("Task goal (not an executable action):", 1)[1].split("\n\n", 1)[0].strip()
        observation = user.split("Current observation:\n", 1)[1].split("\n\nCurrent admissible actions", 1)[0]
        observation = observation.split("\n\nRecent action/tool history:", 1)[0]
    else:
        mission_match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", user)
        mission = mission_match.group(1).strip() if mission_match else "Complete the ALFWorld task."
        if "Your current observation is:" in user:
            observation = user.split("Your current observation is:", 1)[1].split("\nYour admissible actions", 1)[0].strip()
        else:
            observation = user.split("Current observation:", 1)[1].split("\n\nRecent action/tool history:", 1)[0]
            observation = observation.split("\n\nCurrent admissible actions", 1)[0].strip()
    messages = [{"role": "user", "content": build_user_prompt(
        mission=mission, observation=observation, admissible_actions=admissible_actions,
        enable_thinking=thinking, avoid_repeated_actions=avoid_repeated,
    )}]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
        avoid_repeated_actions=avoid_repeated,
        chat_template=QWEN3_ALFWORLD_CHAT_TEMPLATE,
    )
    result: dict[str, object] = {
        "profile": args.profile,
        "model": str(model_path),
        "template": profile["template"],
        "sample_id": row["extra_info"]["sample_id"],
        "game_file": row["extra_info"]["game_file"],
        "rendered_prompt": rendered,
        "rendered_prompt_chars": len(rendered),
        "rendered_prompt_tokens": len(tokenizer(rendered, add_special_tokens=False)["input_ids"]),
        "prompt_contract": {
            "version": PROMPT_VERSION if thinking else NONTHINKING_PROMPT_VERSION,
            "enable_thinking": thinking,
            "required_function": "alfworld_action",
            "required_parameter": "action",
            "output_format": "Qwen3 XML tool call",
            "runtime_termination_ignored": list(RUNTIME_TERMINATION_MARKERS),
        },
        "admissible_action_count": len(admissible_actions),
        "generation": [],
    }
    if not args.no_generate:
        import torch
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model_cls = AutoModelForCausalLM if getattr(model_config, "model_type", "") == "qwen2" else AutoModelForImageTextToText
        model = model_cls.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=args.device if torch.cuda.is_available() else "cpu",
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, do_sample=True, temperature=1.0, top_p=1.0,
                num_return_sequences=args.samples, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generations = []
        for seq in generated_ids[:, prompt_len:]:
            text = tokenizer.decode(seq, skip_special_tokens=False)
            decision = parse_xml_decision(text, enable_thinking=thinking)
            status, reason = decision.status, decision.reason
            if decision.status == "valid" and decision.action not in admissible_actions:
                status, reason = "invalid_action", "inadmissible_action"
            generations.append({"class": status, "strict_xml": decision.status == "valid",
                                "parser_status": decision.status, "validation_status": status,
                                "reason": reason, "action": decision.action, "text": text})
        result["generation"] = generations
        result["generation_class_counts"] = dict(Counter(x["class"] for x in generations))
        result["parser_status_counts"] = dict(Counter(x["parser_status"] for x in generations))
        result["validation_status_counts"] = dict(Counter(x["validation_status"] for x in generations))
        result["valid_action_rate"] = sum(x["class"] == "valid" for x in generations) / len(generations) if generations else 0.0
        result["strict_xml_rate"] = sum(x["strict_xml"] for x in generations) / len(generations) if generations else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("sample_id", "rendered_prompt_chars", "rendered_prompt_tokens", "generation_class_counts", "parser_status_counts") if k in result}, ensure_ascii=False))
    print(f"diagnostic_saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
