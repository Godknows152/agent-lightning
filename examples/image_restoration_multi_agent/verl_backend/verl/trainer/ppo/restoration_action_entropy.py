# Copyright 2026 Microsoft Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Legal restoration-action distributions at the thinking decision point."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import torch

# The current decision-point detector selects the first tool name in the first
# thinking block. ``stop`` is forbidden on that turn, so this is a 16-way space.
FIRST_TURN_RESTORATION_ACTIONS: tuple[str, ...] = (
    "real_esrgan",
    "scunet",
    "retinexformer_fivek",
    "hvicidnet",
    "lightdiff",
    "turbo_rain",
    "s2former",
    "idt",
    "ridcp",
    "kanet",
    "turbo_snow",
    "snowmaster",
    "nafnet_denoise",
    "focalnet_dehaze",
    "focalnet_desnow",
    "mb_taylorformer_dehaze",
)

ALL_TURN_RESTORATION_ACTIONS: tuple[str, ...] = (*FIRST_TURN_RESTORATION_ACTIONS, "stop")

# Exact model-facing action names used by the 0731 SFT thinking targets. Sequence
# probabilities are assigned to these complete strings, not just their first
# token and not the canonical tool-schema values.
SFT_THINKING_ACTION_SURFACES: Mapping[str, str] = {
    "real_esrgan": "A_real_esrgan",
    "scunet": "B_scunet",
    "retinexformer_fivek": "C_retinexformer_fivek",
    "hvicidnet": "D_hvicidnet",
    "lightdiff": "E_lightdiff",
    "turbo_rain": "F_turbo_rain",
    "s2former": "G_s2former",
    "idt": "H_idt",
    "ridcp": "I_ridcp",
    "kanet": "J_kanet",
    "turbo_snow": "K_turbo_snow",
    "snowmaster": "L_snowmaster",
    "nafnet_denoise": "M_nafnet_denoise",
    "focalnet_dehaze": "N_focalnet_dehaze",
    "focalnet_desnow": "O_focalnet_desnow",
    "mb_taylorformer_dehaze": "P_mb_taylorformer_dehaze",
}

ALL_TURN_ACTION_SURFACES: Mapping[str, str] = {
    **SFT_THINKING_ACTION_SURFACES,
    "stop": "stop",
}

if set(SFT_THINKING_ACTION_SURFACES) != set(FIRST_TURN_RESTORATION_ACTIONS):
    raise RuntimeError("SFT thinking surfaces must cover every first-turn restoration action exactly once")


def action_match_variants(action: str) -> tuple[str, ...]:
    """Return stable thinking-text aliases for one canonical action."""

    return (SFT_THINKING_ACTION_SURFACES[action],)


def find_first_restoration_action(text: str) -> tuple[int, str] | None:
    """Find the earliest legal action surface, case-insensitively."""

    lower_text = text.lower()
    best: tuple[int, str] | None = None
    for action in FIRST_TURN_RESTORATION_ACTIONS:
        for variant in action_match_variants(action):
            position = lower_text.find(variant.lower())
            if position != -1 and (best is None or position < best[0]):
                best = (position, action)
    return best


def find_called_restoration_action(
    text: str,
    *,
    legal_actions: Sequence[str] = ALL_TURN_RESTORATION_ACTIONS,
    action_surfaces: Mapping[str, str] = ALL_TURN_ACTION_SURFACES,
) -> str | None:
    """Return the exact model-facing action copied into the XML tool call."""

    surface_to_action = {action_surfaces[action].casefold(): action for action in legal_actions}
    matches = re.findall(
        r"<parameter\s*=\s*action\s*>\s*([^<]+?)\s*</parameter\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for value in matches:
        action = surface_to_action.get(value.strip().casefold())
        if action is not None:
            return action
    return None


def find_restoration_decision_in_assistant_turn(
    text: str,
    *,
    legal_actions: Sequence[str],
    action_surfaces: Mapping[str, str] = ALL_TURN_ACTION_SURFACES,
) -> tuple[int, str] | None:
    """Locate the called action's first occurrence in one assistant thinking block."""

    think_end = text.find("</think>")
    if think_end == -1:
        return None

    action = find_called_restoration_action(
        text,
        legal_actions=legal_actions,
        action_surfaces=action_surfaces,
    )
    if action is None:
        return None

    surface = action_surfaces[action]
    match = re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(surface)}(?![A-Za-z0-9_])",
        text[:think_end],
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.start(), action


def tokenize_action_surfaces(
    tokenizer,
    *,
    leading_text: str,
    actions: Sequence[str] = FIRST_TURN_RESTORATION_ACTIONS,
    action_surfaces: Mapping[str, str] = SFT_THINKING_ACTION_SURFACES,
) -> list[list[int]]:
    """Tokenize every complete legal action from one exact token boundary."""

    if not actions:
        raise ValueError("actions must not be empty")
    if set(actions) != set(action_surfaces):
        raise ValueError("action_surfaces must cover the requested actions exactly once")

    tokenized_actions = []
    for action in actions:
        token_ids = tokenizer.encode(leading_text + action_surfaces[action], add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"tokenizer produced no tokens for action {action!r}")
        tokenized_actions.append([int(token_id) for token_id in token_ids])
    return tokenized_actions


def tokenize_action_first_tokens(
    tokenizer,
    *,
    leading_text: str,
    actions: Sequence[str],
    action_surfaces: Mapping[str, str] = ALL_TURN_ACTION_SURFACES,
) -> list[int]:
    """Tokenize context-specific first branches for every legal action."""

    if len(actions) < 2:
        raise ValueError("actions must contain at least two legal choices")

    first_token_ids = []
    for action in actions:
        if action not in action_surfaces:
            raise ValueError(f"missing model-facing surface for action {action!r}")
        token_ids = tokenizer.encode(leading_text + action_surfaces[action], add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"tokenizer produced no tokens for action {action!r}")
        first_token_ids.append(int(token_ids[0]))
    return first_token_ids


def action_sequence_entropies(sequence_log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and normalized categorical entropy over complete action strings.

    ``sequence_log_probs[..., a]`` must be the sum of teacher-forced token log
    probabilities for action ``a``. All operations remain in the autograd graph.
    """

    if sequence_log_probs.ndim < 1 or sequence_log_probs.shape[-1] < 2:
        raise ValueError("sequence_log_probs must contain at least two actions")

    scores = sequence_log_probs.float()
    normalized_log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    probabilities = normalized_log_probs.exp()
    entropy = -(probabilities * normalized_log_probs).sum(dim=-1)
    normalized_entropy = entropy / math.log(sequence_log_probs.shape[-1])
    return entropy, normalized_entropy


def first_token_action_entropies(
    action_logits: torch.Tensor,
    legal_action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and normalized entropy over legal action first-token logits."""

    if action_logits.ndim < 1 or action_logits.shape[-1] < 2:
        raise ValueError("action_logits must contain at least two action slots")

    scores = action_logits.float()
    if legal_action_mask is None:
        legal_action_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        legal_action_mask = legal_action_mask.to(device=scores.device, dtype=torch.bool)
        if legal_action_mask.shape != scores.shape:
            raise ValueError("legal_action_mask must have the same shape as action_logits")

    action_counts = legal_action_mask.sum(dim=-1)
    if (action_counts < 2).any():
        raise ValueError("every decision point must contain at least two legal actions")

    masked_scores = scores.masked_fill(~legal_action_mask, torch.finfo(scores.dtype).min)
    log_probs = torch.nn.functional.log_softmax(masked_scores, dim=-1)
    probabilities = log_probs.exp()
    entropy_terms = torch.where(legal_action_mask, probabilities * log_probs, torch.zeros_like(log_probs))
    entropy = -entropy_terms.sum(dim=-1)
    normalized_entropy = entropy / action_counts.to(torch.float32).log()
    return entropy, normalized_entropy
