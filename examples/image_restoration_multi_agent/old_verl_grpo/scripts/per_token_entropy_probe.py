"""Probe per-token entropy under the first-turn inputs used by old-VERL GRPO.

The probe reports two different distributions:

* ``policy_entropy`` is full-vocabulary entropy after temperature scaling. This
  matches the entropy recomputed by the actor and used by entropy regularization.
* ``sampling_entropy`` is entropy after temperature, multimodal-token bias,
  top-k, and top-p. This is the distribution sampled by this standalone probe.
* ``action_sequence_entropy`` is categorical entropy over the normalized
  probabilities of the complete first-turn legal model-facing action strings.
  The 0731 action vocabulary also gives every legal action a distinct first
  branch token.

The prompt, tool schema, image bytes, and multimodal message structure come from
the same repository sources as ``ToolAgentLoop``. No temporary prompt/schema
snapshots are used.
"""

# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_PATH = Path(__file__).resolve()
OLD_VERL_ROOT = SCRIPT_PATH.parents[1]
IMAGE_RESTORATION_ROOT = SCRIPT_PATH.parents[2]
VERL_BACKEND_ROOT = IMAGE_RESTORATION_ROOT / "verl_backend"
LOCAL_PYDEPS_ROOT = OLD_VERL_ROOT / ".pydeps"

# The training launcher prepends .pydeps because it contains the Qwen3.5
# Transformers/SGLang support that is not available in the base conda package.
# Do this before importing transformers so direct script execution resolves the
# same implementation as GRPO.
for import_root in (LOCAL_PYDEPS_ROOT, IMAGE_RESTORATION_ROOT, VERL_BACKEND_ROOT):
    if import_root == LOCAL_PYDEPS_ROOT and not import_root.is_dir():
        continue
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from agents.prompts import (  # noqa: E402
    build_expert_single_step_sft_system_prompt,
    build_expert_single_step_sft_user_prompt,
)
from schemas import ExpertName  # noqa: E402
from tool_registry import ToolRegistry  # noqa: E402
from verl.trainer.ppo.restoration_action_entropy import (  # noqa: E402
    FIRST_TURN_RESTORATION_ACTIONS as DECISION_POINT_ACTIONS,
    SFT_THINKING_ACTION_SURFACES,
    action_match_variants,
)
from verl.utils.chat_template import apply_chat_template  # noqa: E402
from verl.workers.rollout.utils import get_multimodal_special_token_ids  # noqa: E402

DEFAULT_MODEL_PATH = "/home/LXJ/Python_Projects/Models/Qwen3.5-9B"
DEFAULT_LORA_PATH = (
    "/home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_experts/outputs/"
    "qwen3_5_0731/format_cold_start/fog"
)
DEFAULT_DATA_PATH = str(OLD_VERL_ROOT / "data" / "fog_train.parquet")
DEFAULT_TOOL_REGISTRY_PATH = str(IMAGE_RESTORATION_ROOT / "config" / "tools.yaml")
DEFAULT_OUTPUT_DIR = str(OLD_VERL_ROOT / "outputs" / "per_token_entropy_analysis_aligned")
GRPO_STOP_TOKEN_IDS = (248046, 248044)

EXPERT_ALIASES = {
    "fog": ExpertName.FOG,
    "fog_expert": ExpertName.FOG,
    "rain": ExpertName.RAIN,
    "rain_expert": ExpertName.RAIN,
    "snow": ExpertName.SNOW,
    "snow_expert": ExpertName.SNOW,
    "lowlight": ExpertName.LOW_LIGHT,
    "low_light": ExpertName.LOW_LIGHT,
    "low-light": ExpertName.LOW_LIGHT,
    "low_light_expert": ExpertName.LOW_LIGHT,
}


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Compute categorical entropy in FP32 to avoid BF16 cancellation."""

    logits_fp32 = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits_fp32, dim=-1)
    probabilities = log_probs.exp()
    return float(torch.sum(-probabilities * log_probs, dim=-1).item())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",")]


def resolve_expert_name(row: pd.Series) -> ExpertName:
    """Resolve the expert exactly as ``ToolAgentLoop`` does."""

    extra_info = _as_dict(row.get("extra_info"))
    raw_expert = str(extra_info.get("expert_name") or extra_info.get("expert") or "fog")
    if raw_expert in EXPERT_ALIASES:
        return EXPERT_ALIASES[raw_expert]
    return ExpertName(raw_expert)


def load_parquet_image(row: pd.Series) -> tuple[Image.Image, str]:
    """Load the exact image payload consumed by ``RLHFDataset``."""

    image_entries = row.get("images")
    if image_entries is not None and len(image_entries) > 0:
        image_entry = image_entries[0]
        if isinstance(image_entry, dict) and image_entry.get("bytes") is not None:
            return Image.open(BytesIO(image_entry["bytes"])).convert("RGB"), "parquet:images[0].bytes"
        if isinstance(image_entry, dict) and image_entry.get("path"):
            image_path = str(image_entry["path"])
            return Image.open(image_path).convert("RGB"), image_path

    image_path = _as_dict(row.get("extra_info")).get("image_path")
    if not image_path:
        raise ValueError("sample has neither images[0].bytes nor extra_info.image_path")
    return Image.open(image_path).convert("RGB"), str(image_path)


def limit_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Match ``AgentLoopBase._limit_image_pixels`` for Qwen3.5 inputs."""

    if max_pixels <= 0:
        raise ValueError(f"max_image_pixels must be positive, got {max_pixels}")
    width, height = image.size
    if width * height <= max_pixels:
        return image

    scale = (max_pixels / float(width * height)) ** 0.5
    target_width = max(32, int(width * scale) // 32 * 32)
    target_height = max(32, int(height * scale) // 32 * 32)
    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > 32:
            target_width -= 32
        elif target_height > 32:
            target_height -= 32
        else:
            break
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def build_grpo_initial_context(
    row: pd.Series,
    registry: ToolRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the same first-turn messages and schema as ``ToolAgentLoop``."""

    expert_name = resolve_expert_name(row)
    user_text = build_expert_single_step_sft_user_prompt().removeprefix("<image>\n")
    messages = [
        {
            "role": "system",
            "content": build_expert_single_step_sft_system_prompt(expert_name, registry),
        },
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        },
    ]
    return messages, [registry.build_tool_schema(include_stop=True)]


def prepare_grpo_inputs(
    processor: Any,
    row: pd.Series,
    registry: ToolRegistry,
    *,
    max_image_pixels: int,
    device: Optional[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create the same first-turn multimodal model inputs as GRPO."""

    messages, tools = build_grpo_initial_context(row, registry)
    image, image_source = load_parquet_image(row)
    image = limit_image_pixels(image, max_image_pixels)
    raw_prompt = apply_chat_template(
        processor,
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[raw_prompt],
        images=[image],
        return_tensors="pt",
        do_sample_frames=False,
    )
    if device is not None:
        model_inputs = model_inputs.to(device)

    prompt_ids = model_inputs["input_ids"][0].detach().cpu().numpy().astype(np.int64, copy=False)
    decoded_prompt = processor.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
    multimodal_token_ids = set(get_multimodal_special_token_ids(processor))
    present_multimodal_ids = sorted(multimodal_token_ids.intersection(prompt_ids.tolist()))
    if not present_multimodal_ids:
        raise RuntimeError(
            "GRPO prompt contains no multimodal special token; the image would not be visible to the model"
        )

    action_enum = tools[0]["function"]["parameters"]["properties"]["action"]["enum"]
    extra_info = _as_dict(row.get("extra_info"))
    metadata = {
        "prompt_length": int(prompt_ids.size),
        "prompt_sha256": hashlib.sha256(prompt_ids.tobytes()).hexdigest(),
        "prompt_text": raw_prompt,
        "decoded_prompt": decoded_prompt,
        "decoded_prompt_sha256": hashlib.sha256(decoded_prompt.encode("utf-8")).hexdigest(),
        "multimodal_token_ids": present_multimodal_ids,
        "image_source": image_source,
        "image_size": list(image.size),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "sample_id": extra_info.get("sample_id"),
        "expert": resolve_expert_name(row).value,
        "tool_schema_includes_stop": "stop" in action_enum,
    }
    return dict(model_inputs), metadata


def apply_sampling_filters(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    blocked_token_ids: Sequence[int],
) -> torch.Tensor:
    """Build the filtered categorical distribution used for sampling."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    processed_logits = logits.float().clone()
    valid_blocked_ids = [token_id for token_id in blocked_token_ids if 0 <= token_id < processed_logits.numel()]
    if valid_blocked_ids:
        processed_logits[valid_blocked_ids] += -100.0
    processed_logits /= temperature

    if top_k > 0 and top_k < processed_logits.numel():
        cutoff = torch.topk(processed_logits, top_k).values[-1]
        processed_logits = processed_logits.masked_fill(processed_logits < cutoff, float("-inf"))

    probabilities = torch.nn.functional.softmax(processed_logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        probabilities[sorted_indices[remove_sorted]] = 0.0
        probabilities /= probabilities.sum()
    return probabilities


def extend_sequence_inputs_for_generated_token(generation_inputs: dict[str, torch.Tensor]) -> None:
    """Extend Qwen sequence metadata after appending one visible text token."""

    for key, fill_value in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
        if key not in generation_inputs:
            continue
        sequence_values = generation_inputs[key]
        generation_inputs[key] = torch.cat(
            [
                sequence_values,
                torch.full(
                    (sequence_values.shape[0], 1),
                    fill_value,
                    device=sequence_values.device,
                    dtype=sequence_values.dtype,
                ),
            ],
            dim=1,
        )


def build_inputs_with_visible_tokens(
    inputs: dict[str, torch.Tensor],
    appended_token_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Append visible response tokens while preserving Qwen multimodal metadata."""

    extended_inputs = dict(inputs)
    input_ids = extended_inputs["input_ids"]
    if appended_token_ids:
        appended_ids = torch.tensor(
            [list(appended_token_ids)],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        extended_inputs["input_ids"] = torch.cat([input_ids, appended_ids], dim=1)

        append_length = appended_ids.shape[1]
        for key, fill_value in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
            if key not in extended_inputs:
                continue
            sequence_values = extended_inputs[key]
            extended_inputs[key] = torch.cat(
                [
                    sequence_values,
                    torch.full(
                        (sequence_values.shape[0], append_length),
                        fill_value,
                        device=sequence_values.device,
                        dtype=sequence_values.dtype,
                    ),
                ],
                dim=1,
            )
    return extended_inputs


def normalize_action_sequence_logprobs(sequence_logprobs: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Normalize full action-sequence scores and return their categorical entropy."""

    if sequence_logprobs.ndim != 1 or sequence_logprobs.numel() == 0:
        raise ValueError("sequence_logprobs must be a non-empty one-dimensional tensor")
    normalized_logprobs = sequence_logprobs.float() - torch.logsumexp(sequence_logprobs.float(), dim=0)
    probabilities = normalized_logprobs.exp()
    entropy = float(torch.sum(-probabilities * normalized_logprobs).item())
    return probabilities, entropy


def score_action_sequence_distribution(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    response_prefix_ids: Sequence[int],
    *,
    candidate_leading_text: str,
    actions: Sequence[str],
    action_surfaces: dict[str, str],
    temperature: float,
    selected_action: Optional[str] = None,
) -> dict[str, Any]:
    """Score complete legal action strings from one fixed decision prefix.

    The returned entropy is categorical entropy over full action-string sequence
    probabilities. It is not the entropy of the first token and not a sum of
    per-token entropies.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not actions:
        raise ValueError("actions must not be empty")
    if set(actions) != set(action_surfaces):
        raise ValueError("action_surfaces must cover the requested actions exactly once")

    prompt_length = int(inputs["input_ids"].shape[1])
    decision_prefix_length = prompt_length + len(response_prefix_ids)
    action_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for action in actions:
            surface_form = action_surfaces[action]
            scored_text = candidate_leading_text + surface_form
            action_token_ids = tokenizer.encode(scored_text, add_special_tokens=False)
            if not action_token_ids:
                raise RuntimeError(f"tokenizer produced no tokens for action {action!r}")

            appended_ids = [*response_prefix_ids, *action_token_ids]
            scoring_inputs = build_inputs_with_visible_tokens(inputs, appended_ids)
            outputs = model(**scoring_inputs, use_cache=False)

            first_prediction_position = decision_prefix_length - 1
            final_prediction_position = first_prediction_position + len(action_token_ids)
            candidate_logits = outputs.logits[
                0,
                first_prediction_position:final_prediction_position,
                :,
            ].float()
            if candidate_logits.shape[0] != len(action_token_ids):
                raise RuntimeError(
                    f"expected {len(action_token_ids)} prediction positions for {action!r}, "
                    f"got {candidate_logits.shape[0]}"
                )

            candidate_logprobs = torch.nn.functional.log_softmax(candidate_logits / temperature, dim=-1)
            target_ids = torch.tensor(action_token_ids, device=candidate_logprobs.device, dtype=torch.long)
            token_logprobs = candidate_logprobs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            sequence_logprob = float(token_logprobs.sum().item())
            action_rows.append(
                {
                    "action": action,
                    "surface_form": surface_form,
                    "scored_text": scored_text,
                    "token_ids": [int(token_id) for token_id in action_token_ids],
                    "token_texts": [tokenizer.decode([token_id]) for token_id in action_token_ids],
                    "token_logprobs": [float(value) for value in token_logprobs.detach().cpu().tolist()],
                    "sequence_logprob": sequence_logprob,
                    "mean_token_logprob": sequence_logprob / len(action_token_ids),
                    "first_token_id": int(action_token_ids[0]),
                    "first_token_text": tokenizer.decode([action_token_ids[0]]),
                    "first_token_logprob": float(token_logprobs[0].item()),
                }
            )
            del outputs, candidate_logits, candidate_logprobs, token_logprobs

    sequence_logprob_tensor = torch.tensor(
        [row["sequence_logprob"] for row in action_rows],
        dtype=torch.float32,
    )
    action_probabilities, action_entropy = normalize_action_sequence_logprobs(sequence_logprob_tensor)
    for row, probability in zip(action_rows, action_probabilities.tolist(), strict=True):
        row["normalized_probability"] = float(probability)

    action_rows.sort(key=lambda row: row["normalized_probability"], reverse=True)
    for rank, row in enumerate(action_rows, 1):
        row["rank"] = rank

    first_token_groups: dict[int, dict[str, Any]] = {}
    grouped_actions: defaultdict[int, list[str]] = defaultdict(list)
    grouped_logprobs: dict[int, float] = {}
    for row in action_rows:
        token_id = row["first_token_id"]
        grouped_actions[token_id].append(row["action"])
        grouped_logprobs[token_id] = row["first_token_logprob"]
    unique_first_token_ids = sorted(grouped_actions)
    first_token_logprob_tensor = torch.tensor(
        [grouped_logprobs[token_id] for token_id in unique_first_token_ids],
        dtype=torch.float32,
    )
    first_token_probabilities, first_token_group_entropy = normalize_action_sequence_logprobs(
        first_token_logprob_tensor
    )
    for token_id, probability in zip(unique_first_token_ids, first_token_probabilities.tolist(), strict=True):
        first_token_groups[token_id] = {
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "actions": sorted(grouped_actions[token_id]),
            "policy_logprob": grouped_logprobs[token_id],
            "normalized_probability": float(probability),
        }

    action_count = len(action_rows)
    maximum_entropy = math.log(action_count)
    selected_row = next((row for row in action_rows if row["action"] == selected_action), None)
    return {
        "definition": "categorical entropy over normalized complete action-string sequence probabilities",
        "temperature": float(temperature),
        "surface_form_source": "0731 first-token-distinct SFT thinking targets",
        "candidate_leading_text": candidate_leading_text,
        "action_count": action_count,
        "maximum_entropy": maximum_entropy,
        "sequence_entropy": action_entropy,
        "normalized_sequence_entropy": action_entropy / maximum_entropy if maximum_entropy > 0 else 0.0,
        "candidate_sequence_log_mass": float(torch.logsumexp(sequence_logprob_tensor, dim=0).item()),
        "actions": action_rows,
        "first_token_group_count": len(first_token_groups),
        "first_token_group_entropy": first_token_group_entropy,
        "normalized_first_token_group_entropy": (
            first_token_group_entropy / maximum_entropy if maximum_entropy > 0 else 0.0
        ),
        "first_token_groups": sorted(
            first_token_groups.values(),
            key=lambda group: group["normalized_probability"],
            reverse=True,
        ),
        "shared_first_token_groups": [
            group
            for group in first_token_groups.values()
            if len(group["actions"]) > 1
        ],
        "selected_action": selected_action,
        "selected_action_probability": (
            selected_row["normalized_probability"] if selected_row is not None else None
        ),
        "selected_action_rank": selected_row["rank"] if selected_row is not None else None,
    }


def generate_with_per_token_entropy(
    model: Any,
    processor: Any,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 2500,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    blocked_token_ids: Sequence[int] = (),
    stop_token_ids: Sequence[int] = GRPO_STOP_TOKEN_IDS,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate one first-turn response with GRPO-equivalent constraints."""

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    # Do not mutate the processor output supplied by the caller.  In
    # particular, the attention mask grows with every generated token while
    # the multimodal tensors remain fixed to the original image.
    generation_inputs = dict(inputs)
    generated_ids = generation_inputs["input_ids"].clone()
    prompt_len = generated_ids.shape[1]
    tokens: list[int] = []
    token_texts: list[str] = []
    raw_entropies: list[float] = []
    policy_entropies: list[float] = []
    sampling_entropies: list[float] = []
    policy_logprobs: list[float] = []
    sampling_logprobs: list[float] = []
    sampled_probs: list[float] = []

    generator = torch.Generator(device=generated_ids.device)
    generator.manual_seed(seed)
    stop_token_id_set = {int(token_id) for token_id in stop_token_ids}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                **{key: value for key, value in generation_inputs.items() if key != "input_ids"},
                input_ids=generated_ids,
                use_cache=False,
            )
            raw_logits = outputs.logits[0, -1, :].float()
            policy_logits = raw_logits / temperature
            sampling_probs = apply_sampling_filters(
                raw_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                blocked_token_ids=blocked_token_ids,
            )
            next_token_id = int(torch.multinomial(sampling_probs, 1, generator=generator).item())

            sampling_log_probs = torch.log(sampling_probs.clamp_min(torch.finfo(torch.float32).tiny))
            policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            raw_entropies.append(entropy_from_logits(raw_logits))
            policy_entropies.append(entropy_from_logits(policy_logits))
            sampling_entropies.append(float(torch.sum(-sampling_probs * sampling_log_probs).item()))
            policy_logprobs.append(float(policy_log_probs[next_token_id].item()))
            sampling_logprobs.append(float(sampling_log_probs[next_token_id].item()))
            sampled_probs.append(float(sampling_probs[next_token_id].item()))
            tokens.append(next_token_id)
            token_texts.append(processor.tokenizer.decode([next_token_id]))

            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[next_token_id]], device=generated_ids.device)],
                dim=1,
            )
            # Qwen3.5's processor returns both attention_mask and
            # mm_token_type_ids as sequence-aligned tensors.  New response
            # tokens are visible text tokens, so append 1 to the former and 0
            # to the latter on every decoding step.
            extend_sequence_inputs_for_generated_token(generation_inputs)

            if next_token_id in stop_token_id_set:
                break
            if "</tool_call>" in processor.tokenizer.decode(tokens, skip_special_tokens=False):
                break

    # SGLang includes matched stop tokens, while ToolAgentLoop removes terminal
    # EOS/PAD before exposing the model response.
    while tokens and tokens[-1] in stop_token_id_set:
        tokens.pop()
        token_texts.pop()
        raw_entropies.pop()
        policy_entropies.pop()
        sampling_entropies.pop()
        policy_logprobs.pop()
        sampling_logprobs.pop()
        sampled_probs.pop()

    return {
        "tokens": tokens,
        "token_texts": token_texts,
        "raw_entropies": raw_entropies,
        "policy_entropies": policy_entropies,
        "sampling_entropies": sampling_entropies,
        "policy_logprobs": policy_logprobs,
        "sampling_logprobs": sampling_logprobs,
        "probs": sampled_probs,
        "full_text": processor.tokenizer.decode(tokens, skip_special_tokens=False),
        "prompt_len": prompt_len,
    }


def find_decision_point(
    full_text: str,
    token_ids: Sequence[int],
    tokenizer: Any,
    actions: Sequence[str] = DECISION_POINT_ACTIONS,
) -> tuple[int, Optional[str], str]:
    """Match the training heuristic and return the token-boundary continuation prefix."""

    think_end = full_text.find("</think>")
    if think_end == -1:
        return -1, None, ""
    thinking_text = full_text[:think_end]
    lower_thinking = thinking_text.lower()

    best: Optional[tuple[int, str]] = None
    for action in actions:
        for variant in action_match_variants(action):
            position = lower_thinking.find(variant.lower())
            if position != -1 and (best is None or position < best[0]):
                best = (position, action)
            if position != -1:
                break
    if best is None:
        return -1, None, ""

    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(token_ids):
        raise ValueError(
            f"decoded response re-tokenized to {len(offsets)} tokens, expected {len(token_ids)}; "
            "cannot align the decision point safely"
        )
    action_char_start, action_name = best
    for token_index, (char_start, char_end) in enumerate(offsets):
        if char_start <= action_char_start < char_end or char_start == action_char_start:
            # The action can start after whitespace inside the decision token.
            # Reuse those leading characters when tokenizing every candidate so
            # each candidate starts at exactly the same model token boundary.
            candidate_leading_text = full_text[char_start:action_char_start]
            return token_index, action_name, candidate_leading_text
    return -1, None, ""


def find_high_entropy_regions(entropies: Sequence[float], threshold: float) -> list[int]:
    """Return positions whose policy entropy is above the requested threshold."""

    return [index for index, entropy in enumerate(entropies) if entropy > threshold]


def plot_entropy_curve(
    result: dict[str, Any],
    decision_point: int,
    high_entropy_regions: Sequence[int],
    output_path: Path,
) -> None:
    """Plot policy entropy, filtered sampling entropy, and sampled-token probability."""

    policy_entropies = result["policy_entropies"]
    sampling_entropies = result["sampling_entropies"]
    fig, (entropy_axis, probability_axis) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    entropy_axis.plot(policy_entropies, "b-", linewidth=1, alpha=0.8, label="policy entropy (actor loss)")
    entropy_axis.plot(
        sampling_entropies,
        "g-",
        linewidth=1,
        alpha=0.65,
        label="sampling entropy (after top-p)",
    )
    entropy_axis.axhline(y=0.001, color="gray", linestyle="--", label="H=0.001")
    entropy_axis.axhline(y=0.1, color="orange", linestyle="--", label="H=0.1")
    if decision_point != -1:
        entropy_axis.axvline(
            x=decision_point,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"decision point (pos={decision_point})",
        )
        entropy_axis.scatter(
            [decision_point],
            [policy_entropies[decision_point]],
            c="red",
            s=100,
            zorder=5,
        )
    if high_entropy_regions:
        entropy_axis.scatter(
            high_entropy_regions,
            [policy_entropies[index] for index in high_entropy_regions],
            c="orange",
            marker="^",
            s=40,
            label=f"high policy entropy (n={len(high_entropy_regions)})",
            zorder=5,
        )
    entropy_axis.set_ylabel("Entropy (nats)", fontsize=12)
    entropy_axis.set_title("GRPO-aligned Per-Token Entropy", fontsize=14, fontweight="bold")
    entropy_axis.legend(loc="upper right")
    entropy_axis.grid(True, alpha=0.3)
    entropy_axis.set_yscale("symlog", linthresh=1e-6)

    probabilities = result["probs"]
    probability_axis.plot(probabilities, "g-", linewidth=1, alpha=0.7)
    probability_axis.axhline(y=0.5, color="gray", linestyle="--", label="P=0.5")
    if decision_point != -1:
        probability_axis.axvline(x=decision_point, color="red", linestyle=":", linewidth=2)
        probability_axis.scatter(
            [decision_point],
            [probabilities[decision_point]],
            c="red",
            s=100,
            zorder=5,
        )
    probability_axis.set_xlabel("Token Position", fontsize=12)
    probability_axis.set_ylabel("P(sampled token)", fontsize=12)
    probability_axis.set_title("Filtered Sampling Probability", fontsize=14, fontweight="bold")
    probability_axis.legend(loc="lower right")
    probability_axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_detailed_report(
    result: dict[str, Any],
    decision_point: int,
    action_name: Optional[str],
    high_entropy_regions: Sequence[int],
    output_path: Path,
    entropy_threshold: float,
) -> None:
    """Write a report that keeps actor and sampler entropy semantics separate."""

    policy_entropies = result["policy_entropies"]
    sampling_entropies = result["sampling_entropies"]
    with open(output_path, "w", encoding="utf-8") as report:
        report.write("GRPO-aligned Per-Token Entropy Analysis Report\n")
        report.write("=" * 80 + "\n\n")
        report.write("【统计信息】\n")
        report.write(f"  生成 tokens 数: {len(policy_entropies)}\n")
        report.write(f"  Policy 熵均值（训练 loss 口径）: {np.mean(policy_entropies):.6f} nats\n")
        report.write(f"  Policy 熵最大值: {np.max(policy_entropies):.6f} nats\n")
        report.write(f"  Sampling 熵均值（top-p 后）: {np.mean(sampling_entropies):.6f} nats\n")
        report.write(f"  Sampling 熵最大值: {np.max(sampling_entropies):.6f} nats\n")
        report.write(f"  Policy 熵 > {entropy_threshold} 的位置数: {len(high_entropy_regions)}\n\n")

        report.write("【决策点】\n")
        if decision_point == -1:
            report.write("  未找到决策点\n\n")
        else:
            report.write(f"  位置: {decision_point}\n")
            report.write(f"  Token: {result['token_texts'][decision_point]!r}\n")
            report.write(f"  Action: {action_name}\n")
            report.write(f"  Policy 熵: {policy_entropies[decision_point]:.6f} nats\n")
            report.write(f"  Sampling 熵: {sampling_entropies[decision_point]:.6f} nats\n")
            report.write(f"  Sampling P: {result['probs'][decision_point]:.6f}\n\n")

            action_analysis = result.get("action_sequence_analysis")
            if action_analysis is not None:
                report.write("【完整动作串分类熵】\n")
                report.write(
                    f"  合法动作数: {action_analysis['action_count']}\n"
                    f"  完整动作串序列熵: {action_analysis['sequence_entropy']:.6f} nats\n"
                    f"  理论最大熵: {action_analysis['maximum_entropy']:.6f} nats\n"
                    f"  归一化序列熵: {action_analysis['normalized_sequence_entropy']:.6f}\n"
                    f"  唯一首 Token 数: {action_analysis['first_token_group_count']}\n"
                    f"  有效首 Token 熵: {action_analysis['first_token_group_entropy']:.6f} nats\n"
                    f"  归一化有效首 Token 熵: "
                    f"{action_analysis['normalized_first_token_group_entropy']:.6f}\n"
                    f"  实际动作分类概率: {action_analysis['selected_action_probability']}\n"
                    f"  实际动作分类排名: {action_analysis['selected_action_rank']}\n"
                    f"  Token 边界前导文本: {action_analysis['candidate_leading_text']!r}\n\n"
                )
                report.write(
                    f"  {'Rank':<6} {'Action':<27} {'Thinking surface':<27} {'Tokens':>8} {'logP(sequence)':>16} "
                    f"{'P(action)':>12}\n"
                )
                report.write("  " + "-" * 108 + "\n")
                for row in action_analysis["actions"]:
                    report.write(
                        f"  {row['rank']:<6} {row['action']:<27} {row['surface_form']:<27} "
                        f"{len(row['token_ids']):>8} "
                        f"{row['sequence_logprob']:>16.6f} {row['normalized_probability']:>12.6f}\n"
                    )
                report.write("\n  共享首 Token 的动作组:\n")
                shared_groups = action_analysis["shared_first_token_groups"]
                if not shared_groups:
                    report.write("    无\n")
                for group in shared_groups:
                    report.write(
                        f"    token={group['token_text']!r} id={group['token_id']}: "
                        f"{', '.join(group['actions'])}\n"
                    )
                report.write("\n")

        report.write("【Top-10 Policy 熵位置】\n")
        report.write(f"  {'Rank':<6} {'Pos':<6} {'Token':<20} {'H_policy':>12} {'H_sample':>12}\n")
        report.write("  " + "-" * 64 + "\n")
        for rank, position in enumerate(np.argsort(policy_entropies)[::-1][:10], 1):
            report.write(
                f"  {rank:<6} {position:<6} {result['token_texts'][position]:<20} "
                f"{policy_entropies[position]:>12.6f} {sampling_entropies[position]:>12.6f}\n"
            )

        report.write("\n【生成文本】\n")
        report.write(result["full_text"] + "\n\n")
        report.write("【完整 Token 序列】\n")
        report.write(
            f"  {'Pos':<6} {'Token':<20} {'H_policy':>12} {'H_sample':>12} "
            f"{'logP_policy':>13} {'logP_sample':>13} {'P_sample':>12}\n"
        )
        report.write("  " + "-" * 96 + "\n")
        rows = zip(
            result["token_texts"],
            policy_entropies,
            sampling_entropies,
            result["policy_logprobs"],
            result["sampling_logprobs"],
            result["probs"],
            strict=True,
        )
        for index, (token, h_policy, h_sample, logp_policy, logp_sample, probability) in enumerate(rows):
            marker = " <-- 决策点" if index == decision_point else ""
            marker = marker or (" <-- 高熵" if h_policy > entropy_threshold else "")
            report.write(
                f"  {index:<6} {token:<20} {h_policy:>12.6f} {h_sample:>12.6f} "
                f"{logp_policy:>13.6f} {logp_sample:>13.6f} {probability:>12.6f}{marker}\n"
            )


def select_samples(df: pd.DataFrame, num_samples: int, sample_indices: Sequence[int]) -> pd.DataFrame:
    """Select explicit positional indices or a deterministic random subset."""

    if sample_indices:
        invalid = [index for index in sample_indices if index < 0 or index >= len(df)]
        if invalid:
            raise IndexError(f"sample indices out of range for dataset of size {len(df)}: {invalid}")
        return df.iloc[list(sample_indices)]
    if num_samples <= 0 or num_samples > len(df):
        raise ValueError(f"num_samples must be in [1, {len(df)}], got {num_samples}")
    return df.sample(n=num_samples, random_state=42)


def write_prompt_manifest(
    processor: Any,
    samples: pd.DataFrame,
    registry: ToolRegistry,
    *,
    max_image_pixels: int,
    expected_prompt_length: Optional[int],
    reference_prompt: Optional[str],
    output_path: Path,
) -> list[dict[str, Any]]:
    """Persist prompt parity evidence before loading the policy model."""

    manifest = []
    for df_index, row in samples.iterrows():
        _, metadata = prepare_grpo_inputs(
            processor,
            row,
            registry,
            max_image_pixels=max_image_pixels,
            device=None,
        )
        metadata["df_index"] = int(df_index)
        if expected_prompt_length is not None and metadata["prompt_length"] != expected_prompt_length:
            raise RuntimeError(
                f"prompt length mismatch for df_index={df_index}: "
                f"got {metadata['prompt_length']}, expected {expected_prompt_length}"
            )
        if reference_prompt is not None:
            metadata["reference_training_prompt_match"] = metadata["decoded_prompt"] == reference_prompt
            if not metadata["reference_training_prompt_match"]:
                raise RuntimeError(
                    f"decoded prompt mismatch for df_index={df_index}; "
                    "the probe no longer matches the supplied GRPO training sample"
                )
        manifest.append(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
    return manifest


def load_reference_training_prompt(path: str) -> str:
    """Load the decoded prompt persisted with one real GRPO trajectory."""

    reference_path = Path(path)
    with reference_path.open("r", encoding="utf-8") as reference_file:
        for line_number, line in enumerate(reference_file, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"missing non-empty prompt at {reference_path}:{line_number}")
            return prompt
    raise ValueError(f"reference training JSONL is empty: {reference_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO-aligned per-token entropy probe")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Base model path")
    parser.add_argument("--lora_path", default=DEFAULT_LORA_PATH, help="LoRA adapter path")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Training data parquet")
    parser.add_argument("--tool_registry_path", default=DEFAULT_TOOL_REGISTRY_PATH, help="Canonical tools.yaml")
    parser.add_argument("--device", default="cuda:3", help="Device for standalone generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--sample_indices", default="", help="Comma-separated parquet positional indices")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=2500, help="First-turn GRPO token budget")
    parser.add_argument("--max_image_pixels", type=int, default=589824, help="Match data.max_image_pixels")
    parser.add_argument("--temperature", type=float, default=1.0, help="Match rollout.temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Match rollout.top_p")
    parser.add_argument("--top_k", type=int, default=-1, help="Match rollout.top_k")
    parser.add_argument("--seed", type=int, default=42, help="Standalone sampling seed")
    parser.add_argument(
        "--stop_token_ids",
        default=",".join(str(token_id) for token_id in GRPO_STOP_TOKEN_IDS),
        help="Comma-separated SGLang stop token IDs",
    )
    parser.add_argument("--entropy_threshold", type=float, default=0.1, help="Policy entropy threshold")
    parser.add_argument(
        "--expected_prompt_length",
        type=int,
        default=None,
        help="Fail unless every prompt has this known GRPO token length",
    )
    parser.add_argument(
        "--reference_training_jsonl",
        default="",
        help="Fail unless the decoded prompt matches the first real GRPO sample in this JSONL",
    )
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        help="Verify GRPO prompts without loading the model",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    registry = ToolRegistry.from_yaml(args.tool_registry_path)
    dataframe = pd.read_parquet(args.data_path)
    samples = select_samples(dataframe, args.num_samples, parse_int_list(args.sample_indices))
    reference_prompt = (
        load_reference_training_prompt(args.reference_training_jsonl) if args.reference_training_jsonl else None
    )

    manifest_path = output_dir / "prompt_manifest.json"
    prompt_manifest = write_prompt_manifest(
        processor,
        samples,
        registry,
        max_image_pixels=args.max_image_pixels,
        expected_prompt_length=args.expected_prompt_length,
        reference_prompt=reference_prompt,
        output_path=manifest_path,
    )
    print(f"GRPO prompt manifest: {manifest_path}")
    for item in prompt_manifest:
        print(
            f"  df_index={item['df_index']} sample_id={item['sample_id']} "
            f"tokens={item['prompt_length']} sha256={item['prompt_sha256'][:12]} "
            f"image={item['image_size']}"
            + (
                f" reference_match={item['reference_training_prompt_match']}"
                if "reference_training_prompt_match" in item
                else ""
            )
        )
    if args.prompt_only:
        print("Prompt-only verification complete; model was not loaded.")
        return

    print("=" * 80)
    print("加载 SFT LoRA 模型...")
    print("=" * 80)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        # ``device_map`` accepts a placement mapping (or one of Accelerate's
        # strategy names), not a CUDA device string such as ``cuda:3``.
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print("模型加载完成")

    blocked_token_ids = get_multimodal_special_token_ids(processor)
    stop_token_ids = parse_int_list(args.stop_token_ids)
    all_results = []
    summary = {
        "total_samples": len(samples),
        "samples_with_high_entropy": 0,
        "decision_points_found": 0,
        "max_policy_entropy": 0.0,
        "mean_policy_entropy": 0.0,
        "mean_sampling_entropy": 0.0,
        "mean_decision_action_sequence_entropy": None,
        "mean_normalized_decision_action_sequence_entropy": None,
        "mean_decision_first_token_group_entropy": None,
        "mean_normalized_decision_first_token_group_entropy": None,
    }
    action_sequence_entropy_sum = 0.0
    normalized_action_sequence_entropy_sum = 0.0
    first_token_group_entropy_sum = 0.0
    normalized_first_token_group_entropy_sum = 0.0
    action_sequence_analysis_count = 0

    for sample_number, (df_index, row) in enumerate(samples.iterrows(), 1):
        print("\n" + "=" * 80)
        print(f"样本 {sample_number}/{len(samples)} (df_index={df_index})")
        inputs, prompt_metadata = prepare_grpo_inputs(
            processor,
            row,
            registry,
            max_image_pixels=args.max_image_pixels,
            device=args.device,
        )
        print(
            f"GRPO prompt: tokens={prompt_metadata['prompt_length']} "
            f"sha256={prompt_metadata['prompt_sha256'][:12]} image={prompt_metadata['image_size']}"
        )
        result = generate_with_per_token_entropy(
            model,
            processor,
            inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            blocked_token_ids=blocked_token_ids,
            stop_token_ids=stop_token_ids,
            seed=args.seed + sample_number - 1,
        )
        if not result["tokens"]:
            raise RuntimeError(f"model generated no visible tokens for df_index={df_index}")

        decision_point, action_name, candidate_leading_text = find_decision_point(
            result["full_text"],
            result["tokens"],
            processor.tokenizer,
        )
        high_entropy_positions = find_high_entropy_regions(
            result["policy_entropies"],
            args.entropy_threshold,
        )
        print(f"生成 {len(result['tokens'])} tokens")
        print(f"Policy 平均熵: {np.mean(result['policy_entropies']):.6f} nats")
        print(f"Sampling 平均熵: {np.mean(result['sampling_entropies']):.6f} nats")
        if decision_point == -1:
            print("未找到决策点")
            result["action_sequence_analysis"] = None
        else:
            print(
                f"决策点: pos={decision_point} token={result['token_texts'][decision_point]!r} "
                f"action={action_name} H_policy={result['policy_entropies'][decision_point]:.6f} "
                f"H_sample={result['sampling_entropies'][decision_point]:.6f}"
            )
            print(f"正在计算 {len(DECISION_POINT_ACTIONS)} 个完整合法动作串的序列概率...")
            action_sequence_analysis = score_action_sequence_distribution(
                model,
                processor.tokenizer,
                inputs,
                result["tokens"][:decision_point],
                candidate_leading_text=candidate_leading_text,
                actions=DECISION_POINT_ACTIONS,
                action_surfaces=SFT_THINKING_ACTION_SURFACES,
                temperature=args.temperature,
                selected_action=action_name,
            )
            result["action_sequence_analysis"] = action_sequence_analysis
            print(
                "完整动作串分类熵: "
                f"H={action_sequence_analysis['sequence_entropy']:.6f} "
                f"H_max={action_sequence_analysis['maximum_entropy']:.6f} "
                f"H/H_max={action_sequence_analysis['normalized_sequence_entropy']:.6f} "
                f"H_first/H_max={action_sequence_analysis['normalized_first_token_group_entropy']:.6f} "
                f"首Token组数={action_sequence_analysis['first_token_group_count']}"
            )
            summary["decision_points_found"] += 1
            action_sequence_entropy_sum += action_sequence_analysis["sequence_entropy"]
            normalized_action_sequence_entropy_sum += action_sequence_analysis["normalized_sequence_entropy"]
            first_token_group_entropy_sum += action_sequence_analysis["first_token_group_entropy"]
            normalized_first_token_group_entropy_sum += action_sequence_analysis[
                "normalized_first_token_group_entropy"
            ]
            action_sequence_analysis_count += 1

        if high_entropy_positions:
            summary["samples_with_high_entropy"] += 1
        summary["max_policy_entropy"] = max(
            summary["max_policy_entropy"],
            float(np.max(result["policy_entropies"])),
        )
        summary["mean_policy_entropy"] += float(np.mean(result["policy_entropies"]))
        summary["mean_sampling_entropy"] += float(np.mean(result["sampling_entropies"]))

        sample_dir = output_dir / f"sample_{sample_number:02d}"
        sample_dir.mkdir(exist_ok=True)
        plot_entropy_curve(
            result,
            decision_point,
            high_entropy_positions,
            sample_dir / "entropy_curve.png",
        )
        save_detailed_report(
            result,
            decision_point,
            action_name,
            high_entropy_positions,
            sample_dir / "report.txt",
            args.entropy_threshold,
        )

        sampling_config = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "blocked_token_ids": blocked_token_ids,
            "stop_token_ids": stop_token_ids,
            "seed": args.seed + sample_number - 1,
        }
        with open(sample_dir / "data.json", "w", encoding="utf-8") as data_file:
            json.dump(
                {
                    "df_index": int(df_index),
                    "prompt_metadata": prompt_metadata,
                    "sampling_config": sampling_config,
                    "decision_point": decision_point,
                    "action_name": action_name,
                    "high_entropy_positions": high_entropy_positions,
                    **result,
                },
                data_file,
                indent=2,
                ensure_ascii=False,
            )

        all_results.append(
            {
                "sample": sample_number,
                "df_index": int(df_index),
                "prompt_length": prompt_metadata["prompt_length"],
                "decision_point": decision_point,
                "action": action_name,
                "decision_policy_entropy": (
                    result["policy_entropies"][decision_point] if decision_point != -1 else None
                ),
                "decision_sampling_entropy": (
                    result["sampling_entropies"][decision_point] if decision_point != -1 else None
                ),
                "decision_action_sequence_entropy": (
                    result["action_sequence_analysis"]["sequence_entropy"] if decision_point != -1 else None
                ),
                "normalized_decision_action_sequence_entropy": (
                    result["action_sequence_analysis"]["normalized_sequence_entropy"]
                    if decision_point != -1
                    else None
                ),
                "decision_first_token_group_entropy": (
                    result["action_sequence_analysis"]["first_token_group_entropy"]
                    if decision_point != -1
                    else None
                ),
                "normalized_decision_first_token_group_entropy": (
                    result["action_sequence_analysis"]["normalized_first_token_group_entropy"]
                    if decision_point != -1
                    else None
                ),
                "selected_action_sequence_probability": (
                    result["action_sequence_analysis"]["selected_action_probability"]
                    if decision_point != -1
                    else None
                ),
                "selected_action_sequence_rank": (
                    result["action_sequence_analysis"]["selected_action_rank"]
                    if decision_point != -1
                    else None
                ),
                "mean_policy_entropy": float(np.mean(result["policy_entropies"])),
                "mean_sampling_entropy": float(np.mean(result["sampling_entropies"])),
            }
        )

    summary["mean_policy_entropy"] /= len(samples)
    summary["mean_sampling_entropy"] /= len(samples)
    if action_sequence_analysis_count:
        summary["mean_decision_action_sequence_entropy"] = (
            action_sequence_entropy_sum / action_sequence_analysis_count
        )
        summary["mean_normalized_decision_action_sequence_entropy"] = (
            normalized_action_sequence_entropy_sum / action_sequence_analysis_count
        )
        summary["mean_decision_first_token_group_entropy"] = (
            first_token_group_entropy_sum / action_sequence_analysis_count
        )
        summary["mean_normalized_decision_first_token_group_entropy"] = (
            normalized_first_token_group_entropy_sum / action_sequence_analysis_count
        )
    with open(output_dir / "00_summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write("GRPO-aligned Per-Token Entropy Analysis\n")
        summary_file.write("=" * 80 + "\n\n")
        summary_file.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n\n")
        summary_file.write("各样本：\n")
        for item in all_results:
            summary_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n汇总报告: {output_dir / '00_summary.txt'}")


if __name__ == "__main__":
    main()
