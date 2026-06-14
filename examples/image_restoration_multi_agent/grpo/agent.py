"""Agent Lightning rollout wrapper for one fixed restoration expert."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from agents import ReplayExpertAgent, VLMRestorationExpertAgent
from config import StageGExampleConfig
from controller import ExpertAgent
from factory import RealControllerFactory
from grpo.smoke_runtime import SmokeActionOverrideExpert
from schemas import (
    DEGRADATION_TO_EXPERT,
    DiagnosisResult,
    ExpertName,
    GRPORestorationTask,
    StageFRestorationTask,
    WorkflowResult,
)

import agentlightning as agl
from agentlightning.semconv import LightningSpanAttributes

_AGENT_LIGHTNING_TASK_FIELDS = {"index", "data_id"}


class PromptImageTracingExpert:
    """Attach the current visual observation to the next policy request."""

    def __init__(self, delegate: ExpertAgent) -> None:
        self.delegate = delegate

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        agl.emit_annotation({LightningSpanAttributes.PROMPT_IMAGE_URLS.value: [state.current_image]})
        return self.delegate.decide(state)


def parse_grpo_task(task: dict[str, Any]) -> GRPORestorationTask:
    """Validate business fields while allowing Agent Lightning task metadata."""

    model_fields = set(GRPORestorationTask.model_fields)
    unexpected = set(task) - model_fields - _AGENT_LIGHTNING_TASK_FIELDS
    if unexpected:
        raise ValueError(f"unexpected GRPO task fields: {sorted(unexpected)}")
    return GRPORestorationTask.model_validate({key: task[key] for key in model_fields if key in task})


class GRPOImageRestorationAgent(agl.LitAgent[dict[str, Any]]):
    """Run one oracle-routed expert trajectory and return one scalar reward."""

    def __init__(
        self,
        *,
        config: StageGExampleConfig,
        factory: RealControllerFactory,
        expert_name: ExpertName,
        tokenizer: object,
        smoke_override_actions: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.factory = factory
        self.expert_name = expert_name
        self.tokenizer = tokenizer
        self.smoke_override_actions = smoke_override_actions
        self.results: dict[str, WorkflowResult] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Execute the selected expert without invoking the diagnosis policy."""

        parsed_task = parse_grpo_task(task)
        expected_expert = DEGRADATION_TO_EXPERT[parsed_task.degradation_type]
        if expected_expert != self.expert_name:
            raise ValueError(
                f"{self.expert_name.value} received {parsed_task.degradation_type.value} data "
                f"routed to {expected_expert.value}"
            )
        attempted_rollout = cast(agl.AttemptedRollout, rollout)
        llm = cast(agl.LLM, resources["main_llm"])
        sampling = llm.sampling_parameters
        settings = self.config.expert_vlm.model_copy(
            update={
                "base_url": llm.get_base_url(
                    attempted_rollout.rollout_id,
                    attempted_rollout.attempt.attempt_id,
                ),
                "api_key": llm.api_key or "EMPTY",
                "model": llm.model,
                "temperature": float(sampling.get("temperature", self.config.expert_vlm.temperature)),
                "top_p": float(sampling.get("top_p", self.config.expert_vlm.top_p)),
                "max_tokens": int(sampling.get("max_tokens", self.config.expert_vlm.max_tokens)),
                "seed": int.from_bytes(
                    hashlib.sha256(attempted_rollout.rollout_id.encode("utf-8")).digest()[:4],
                    byteorder="big",
                ),
            }
        )
        resource_name = self.config.expert_resources[self.expert_name].resource_name
        experts = {
            expert: ReplayExpertAgent.from_actions(
                expert,
                ["stop"],
                self.factory.tool_registry,
                resource_name=self.config.expert_resources[expert].resource_name,
            )
            for expert in ExpertName
        }
        selected_expert: ExpertAgent = VLMRestorationExpertAgent(
            settings,
            self.expert_name,
            self.factory.tool_registry,
            max_steps=self.config.workflow.max_steps,
            resource_name=resource_name,
            tokenizer=self.tokenizer,
        )
        if self.smoke_override_actions is not None:
            selected_expert = SmokeActionOverrideExpert(
                selected_expert,
                self.smoke_override_actions,
            )
        selected_expert = PromptImageTracingExpert(selected_expert)
        experts[self.expert_name] = selected_expert

        output_dir = (
            Path(parsed_task.output_root).expanduser().resolve()
            / parsed_task.sample_id
            / attempted_rollout.rollout_id
            / attempted_rollout.attempt.attempt_id
        )
        controller_task = StageFRestorationTask(
            image_path=parsed_task.image_path,
            degradation_type=parsed_task.degradation_type,
            scripted_actions=["stop"],
            output_dir=str(output_dir),
            visual_evidence=parsed_task.visual_evidence,
        )
        diagnosis = DiagnosisResult(
            primary_type=parsed_task.degradation_type,
            visual_evidence=parsed_task.visual_evidence,
            route_to=self.expert_name,
        )
        controller = self.factory.build(
            controller_task,
            routed_expert_override=self.expert_name,
            experts_override=experts,
        )
        result = controller.run(
            controller_task,
            trajectory_id=attempted_rollout.rollout_id,
            trace=True,
            diagnosis_override=diagnosis,
        )
        self.results[attempted_rollout.rollout_id] = result
        if result.state.final_reward is None:
            raise RuntimeError("GRPO controller completed without a trajectory reward")
        return result.state.final_reward
