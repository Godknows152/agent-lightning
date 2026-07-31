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

# Exact action-name surfaces used by the 0721 SFT thinking targets. Sequence
# probabilities are assigned to these complete strings, not just their first
# token and not the canonical tool-schema values.
SFT_THINKING_ACTION_SURFACES: Mapping[str, str] = {
    "real_esrgan": "Real-ESRGAN",
    "scunet": "SCUNet",
    "retinexformer_fivek": "Retinexformer-FiveK",
    "hvicidnet": "HVI-CIDNet",
    "lightdiff": "LightenDiffusion",
    "turbo_rain": "Turbo-Rain",
    "s2former": "S2Former",
    "idt": "IDT",
    "ridcp": "RIDCP",
    "kanet": "KA-Net",
    "turbo_snow": "Turbo-Snow",
    "snowmaster": "SnowMaster",
    "nafnet_denoise": "NAFNet-Denoise",
    "focalnet_dehaze": "FocalNet-Dehaze",
    "focalnet_desnow": "FocalNet-Desnow",
    "mb_taylorformer_dehaze": "MB-TaylorFormer-Dehaze",
}

if set(SFT_THINKING_ACTION_SURFACES) != set(FIRST_TURN_RESTORATION_ACTIONS):
    raise RuntimeError("SFT thinking surfaces must cover every first-turn restoration action exactly once")


def action_match_variants(action: str) -> tuple[str, ...]:
    """Return stable thinking-text aliases for one canonical action."""

    candidates = (
        SFT_THINKING_ACTION_SURFACES[action],
        action,
        action.replace("_", "-"),
        action.replace("_", " "),
    )
    return tuple(dict.fromkeys(candidates))


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
