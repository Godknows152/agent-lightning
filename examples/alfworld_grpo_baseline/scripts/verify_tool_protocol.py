#!/usr/bin/env python3
"""Verify the ALFWorld prompt/template and Qwen2.5 tool-call surface.

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
from alfworld_baseline.prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE
from alfworld_baseline.thinking import tool_output

PROFILES = {
    "qwen25_1_5b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen2.5-1.5B-Instruct"),
        "data": ROOT / "data" / "qwen25_1_5b" / "train.parquet",
        "template": "Qwen2.5 tokenizer native chat_template",
    },
    "qwen35_9b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen3.5-9B"),
        "data": ROOT / "data" / "qwen35_9b" / "train.parquet",
        "template": "Qwen3.5 ALFWorld current-state chat_template",
    },
    "qwen35_2b": {
        "model": Path("/home/LXJ/Python_Projects/Models/Qwen3.5-2B"),
        "data": ROOT / "data" / "qwen35_2b" / "train.parquet",
        "template": "Qwen3.5 ALFWorld current-state chat_template",
    },
}
RUNTIME_TERMINATION_MARKERS = ("<|im_end|>", "<|endoftext|>")
STRICT_XML_RE = re.compile(
    r"^<tool_call>\s*<function=alfworld_action>\s*"
    r"<parameter=action>\s*(.*?)\s*</parameter>\s*"
    r"</function>\s*</tool_call>$",
    re.DOTALL,
)
STRICT_QWEN25_JSON_RE = re.compile(
    r'^<tool_call>\s*\{\s*"name"\s*:\s*"alfworld_action"\s*,\s*'
    r'"arguments"\s*:\s*\{\s*"action"\s*:\s*".*?"\s*\}\s*\}\s*</tool_call>$',
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
    if STRICT_QWEN25_JSON_RE.fullmatch(stripped):
        return "qwen25_json_strict"
    if STRICT_XML_RE.fullmatch(stripped):
        return "qwen3_xml_strict"
    if "<tool_call>" in text and "</tool_call>" in text and "<function=" in text and "<parameter=" in text:
        return "qwen3_xml_complete_with_extra_text"
    if "<tool_call>" in text:
        return "qwen3_xml_incomplete"
    if text.lstrip().startswith("{"):
        return "bare_json"
    return "other"


def classify_visible_response(text: str) -> str:
    """Classify after transport EOS removal (public diagnostic helper)."""

    return classify(text)


def _admissible_actions(messages: list[dict[str, object]]) -> tuple[str, ...]:
    """Extract the action list injected by ``build_user_prompt``."""

    user = next((str(m.get("content", "")) for m in messages if m.get("role") == "user"), "")
    markers = (
        "Current admissible actions (the action value must be copied exactly from this list):\n",
        "Current admissible actions (copy exactly one):\n",
    )
    marker = next((candidate for candidate in markers if candidate in user), None)
    if marker is None:
        return ()
    section = user.split(marker, 1)[1].split("\n\n", 1)[0]
    return tuple(line[2:] for line in section.splitlines() if line.startswith("- "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="qwen35_2b")
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    thinking = args.enable_thinking if args.enable_thinking is not None else args.profile == "qwen35_2b"

    profile = PROFILES[args.profile]
    model_path = profile["model"]
    data_path = profile["data"]
    output_path = ROOT / "outputs" / "diagnostics" / args.profile / "tool_protocol.json"

    row = pd.read_parquet(data_path).iloc[0]
    messages = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
    messages = [dict(x) for x in messages]
    admissible_actions = _admissible_actions(messages)
    from alfworld_baseline.tool_registry import ALFWorldToolRegistry

    if not admissible_actions:
        raise RuntimeError("could not extract admissible actions from the prepared prompt")
    tools = [ALFWorldToolRegistry(admissible_actions).build_tool_schema()]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
        chat_template=QWEN35_ALFWORLD_CHAT_TEMPLATE if args.profile.startswith("qwen35") else None,
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
            "version": row["extra_info"].get("prompt_version"),
            "enable_thinking": thinking,
            "required_function": "alfworld_action",
            "required_parameter": "action",
            "runtime_termination_ignored": list(RUNTIME_TERMINATION_MARKERS),
        },
        "admissible_action_count": len(admissible_actions),
        "generation": [],
    }
    if not args.no_generate:
        import torch
        from alfworld_baseline.parser import parse_tool_call
        from alfworld_baseline.tool_registry import ALFWorldToolRegistry
        from alfworld_baseline.validator import validate_tool_call
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
            executable_text, _ = tool_output(text, enable_thinking=thinking)
            parsed = parse_tool_call(executable_text)
            visible_text, terminal_tokens = strip_runtime_termination(executable_text)
            validation = validate_tool_call(parsed, ALFWorldToolRegistry(admissible_actions)) if admissible_actions else None
            generations.append(
                {
                    "class": classify(executable_text),
                    "strict_xml": classify(executable_text) in {"qwen25_json_strict", "qwen3_xml_strict"},
                    "runtime_termination_tokens": terminal_tokens,
                    "visible_text": visible_text,
                    "parser_status": parsed.status.value,
                    "validation_status": validation.status.value if validation is not None else None,
                    "action": validation.action if validation is not None and validation.is_valid else None,
                    "text": text,
                }
            )
        result["generation"] = generations
        result["generation_class_counts"] = dict(Counter(x["class"] for x in generations))
        result["parser_status_counts"] = dict(Counter(x["parser_status"] for x in generations))
        result["validation_status_counts"] = dict(Counter(x["validation_status"] for x in generations))
        result["strict_xml_rate"] = sum(x["strict_xml"] for x in generations) / len(generations) if generations else 0.0
        result["strict_qwen25_json_rate"] = sum(x["class"] == "qwen25_json_strict" for x in generations) / len(generations) if generations else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("sample_id", "rendered_prompt_chars", "rendered_prompt_tokens", "generation_class_counts", "parser_status_counts") if k in result}, ensure_ascii=False))
    print(f"diagnostic_saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
