"""Agent Lightning rollout wrapper for one fixed restoration expert."""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRetentionConfig:
    """Retention policy for GRPO rollout artifacts."""

    enabled: bool = True
    keep_sample_dirs: int = 256
    min_age_seconds: float = 7200.0
    cleanup_interval_rollouts: int = 48


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


def cleanup_rollout_artifacts(
    output_root: Path,
    *,
    current_sample_dir: Path | None,
    retention: ArtifactRetentionConfig,
) -> int:
    """Remove old rollout sample directories under one expert artifact root."""

    if not retention.enabled or retention.keep_sample_dirs < 0:
        return 0
    root = output_root.expanduser().resolve()
    if not root.is_dir():
        return 0

    lock_file = root / ".artifact_cleanup.lock"
    try:
        import fcntl

        with lock_file.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            return _cleanup_rollout_artifacts_unlocked(root, current_sample_dir, retention)
    except OSError as exc:
        logger.warning("Failed to acquire rollout artifact cleanup lock at %s: %s", lock_file, exc)
        return 0


def _cleanup_rollout_artifacts_unlocked(
    output_root: Path,
    current_sample_dir: Path | None,
    retention: ArtifactRetentionConfig,
) -> int:
    now = time.time()
    current = current_sample_dir.expanduser().resolve() if current_sample_dir is not None else None
    sample_dirs: list[tuple[float, Path]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        resolved = child.resolve()
        if current is not None and resolved == current:
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        sample_dirs.append((mtime, resolved))

    sample_dirs.sort(key=lambda item: item[0], reverse=True)
    delete_candidates = sample_dirs[retention.keep_sample_dirs :]
    removed = 0
    for mtime, sample_dir in delete_candidates:
        if now - mtime < retention.min_age_seconds:
            continue
        try:
            shutil.rmtree(sample_dir)
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to remove old rollout artifact directory %s: %s", sample_dir, exc)
    if removed:
        logger.info("Removed %s old rollout artifact sample directories from %s", removed, output_root)
    return removed


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
        artifact_retention: ArtifactRetentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.factory = factory
        self.expert_name = expert_name
        self.tokenizer = tokenizer
        self.smoke_override_actions = smoke_override_actions
        self.artifact_retention = artifact_retention or ArtifactRetentionConfig()
        self._rollouts_since_cleanup = 0
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
        settings = self.config.expert_vlm.model_copy(
            update={
                "base_url": llm.get_base_url(
                    attempted_rollout.rollout_id,
                    attempted_rollout.attempt.attempt_id,
                ),
                "api_key": llm.api_key or "EMPTY",
                "model": llm.model,
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
            min_stop_tool_calls=self.config.workflow.stop_min_tool_calls,
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
        self._maybe_cleanup_artifacts(Path(parsed_task.output_root), output_dir.parents[1])
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
        agl.emit_annotation(
            {
                "image_restoration.termination_reason": result.state.termination_reason or "unknown",
                "image_restoration.turn_count": len(result.state.steps),
                "image_restoration.final_reward": float(result.state.final_reward),
            }
        )
        return result.state.final_reward

    def _maybe_cleanup_artifacts(self, output_root: Path, current_sample_dir: Path) -> None:
        retention = self.artifact_retention
        if not retention.enabled:
            return
        self._rollouts_since_cleanup += 1
        if self._rollouts_since_cleanup < retention.cleanup_interval_rollouts:
            return
        self._rollouts_since_cleanup = 0
        cleanup_rollout_artifacts(
            output_root,
            current_sample_dir=current_sample_dir,
            retention=retention,
        )
