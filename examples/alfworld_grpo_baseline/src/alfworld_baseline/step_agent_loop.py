"""Opt-in ALFWorld loop for GiGPO-style GRPO step rewards."""

from __future__ import annotations

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState

from .agent_loop import ALFWorldToolAgentLoop
from .step_reward import PENALTY_COUNTERS, STEP_PENALTIES, compute_score


@register("alfworld_step_grpo_agent")
class ALFWorldStepGRPOAgentLoop(ALFWorldToolAgentLoop):
    """Reuse environment/protocol handling, retaining the cost of each decision."""

    ALFWORLD_NO_TOOL_CALL_PENALTY = STEP_PENALTIES["no_action"]
    ALFWORLD_INVALID_TOOL_CALL_PENALTY = STEP_PENALTIES["invalid_action"]
    ALFWORLD_REPEATED_ACTION_PENALTY = STEP_PENALTIES["repeated_action"]

    async def _process_environment_decision(self, agent_data: AgentData) -> AgentState:
        before = {
            kind: int(agent_data.extra_fields.get(field, 0))
            for kind, field in PENALTY_COUNTERS.items()
        }
        state = await super()._process_environment_decision(agent_data)
        deltas = {
            kind: int(agent_data.extra_fields.get(field, 0)) - before[kind]
            for kind, field in PENALTY_COUNTERS.items()
        }
        if (
            any(delta not in (0, 1) for delta in deltas.values())
            or sum(deltas.values()) > 1
        ):
            raise ValueError(
                f"A decision must have at most one ALFWorld penalty: {deltas}"
            )
        kind = next((kind for kind, delta in deltas.items() if delta), "none")
        # Snapshot on the corresponding output, not in the cumulative episode
        # fields. Later decisions/finalization must not overwrite this metadata.
        agent_data.step_outputs[-1].extra_fields.update(
            alfworld_step_penalty_kind=kind,
            alfworld_step_penalty=STEP_PENALTIES[kind],
        )
        return state

    def _finalize_step_outputs(self, data: AgentData) -> list[AgentLoopOutput]:
        extra = dict(data.extra_fields, tool_rewards=list(data.tool_rewards))
        outcome = 10.0 if extra.get("alfworld_terminal_reason") == "success" else 0.0
        penalty_sum = sum(
            output.extra_fields["alfworld_step_penalty"] for output in data.step_outputs
        )
        for index, output in enumerate(data.step_outputs):
            output.extra_fields.update(extra)
            score = compute_score("alfworld", extra_info=output.extra_fields)
            output.reward_score = score
            output.extra_fields.update(
                alfworld_step_index=index,
                alfworld_is_final_step=index == len(data.step_outputs) - 1,
                alfworld_training_backend="gigpo_grpo",
                alfworld_episode_reward=outcome,
                alfworld_episode_penalty_sum=penalty_sum,
                reward_extra_info={"score": score, "episode_reward": outcome},
            )
            if index == len(data.step_outputs) - 1:
                output.metrics = type(output.metrics)(**data.metrics)
        return data.step_outputs
