"""Agent Lightning wrapper for the deterministic restoration controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import ReplayExpertAgent, VLMDegradationDiagnosisAgent, VLMRestorationExpertAgent
from config import StageEExampleConfig, StageFExampleConfig
from controller import ExpertAgent
from factory import DeterministicControllerFactory, RealControllerFactory
from schemas import (
    DEGRADATION_TO_EXPERT,
    DiagnosisResult,
    ExpertDecisionMode,
    ExpertName,
    RealRestorationTask,
    RestorationTask,
    RoutingMode,
    RoutingSource,
    StageEWorkflowResult,
    StageFRestorationTask,
    StageFWorkflowResult,
    VLMDiagnosisAttempt,
    VLMRestorationTask,
    WorkflowResult,
)

import agentlightning as agl


class DeterministicImageRestorationAgent(agl.LitAgent[dict[str, Any]]):
    """Execute one traced deterministic restoration workflow per rollout."""

    def __init__(self, factory: DeterministicControllerFactory) -> None:
        super().__init__()
        self.factory = factory
        self.results: dict[str, WorkflowResult] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Validate the task, run the controller, and return one final reward."""

        del resources
        parsed_task = RestorationTask.model_validate(task)
        controller = self.factory.build(parsed_task)
        result = controller.run(parsed_task, trajectory_id=rollout.rollout_id, trace=True)
        self.results[rollout.rollout_id] = result
        if result.state.final_reward is None:
            raise RuntimeError("controller completed without a final reward")
        return result.state.final_reward


class RealImageRestorationAgent(agl.LitAgent[dict[str, Any]]):
    """Execute one traced stage D workflow with models isolated in `verl`."""

    def __init__(self, factory: RealControllerFactory) -> None:
        super().__init__()
        self.factory = factory
        self.results: dict[str, WorkflowResult] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Validate a real task, run the controller, and return one final reward."""

        del resources
        parsed_task = RealRestorationTask.model_validate(task)
        controller = self.factory.build(parsed_task)
        result = controller.run(parsed_task, trajectory_id=rollout.rollout_id, trace=True)
        self.results[rollout.rollout_id] = result
        if result.state.final_reward is None:
            raise RuntimeError("controller completed without a final reward")
        return result.state.final_reward


class VLMImageRestorationAgent(agl.LitAgent[dict[str, Any]]):
    """Run stage E with one real VLM diagnosis and scripted expert actions."""

    def __init__(
        self,
        config: StageEExampleConfig,
        factory: RealControllerFactory,
        diagnosis_agent: VLMDegradationDiagnosisAgent,
    ) -> None:
        super().__init__()
        self.config = config
        self.factory = factory
        self.diagnosis_agent = diagnosis_agent
        self.results: dict[str, StageEWorkflowResult] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Call the VLM once, resolve routing mode, and run the real workflow."""

        del resources
        parsed_task = VLMRestorationTask.model_validate(task)
        routing_mode = parsed_task.routing_mode or self.config.vlm.routing_mode
        output_dir = Path(parsed_task.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with agl.operation(name="diagnosis_agent.vlm_prediction", **{"agent.name": "diagnosis_agent"}) as operation:
            attempt = self.diagnosis_agent.diagnose(parsed_task.image_path)
            attempt_payload = attempt.model_dump(mode="json")
            operation.set_output(attempt_payload)
            agl.emit_object(attempt_payload, attributes={"restoration.record_type": "vlm_diagnosis_attempt"})

        actual_diagnosis: DiagnosisResult | None
        routing_source: RoutingSource | None
        if routing_mode == RoutingMode.PREDICTED_STRICT:
            actual_diagnosis = attempt.diagnosis
            routing_source = RoutingSource.VLM_PREDICTION if actual_diagnosis is not None else None
        else:
            actual_diagnosis = DiagnosisResult(
                primary_type=parsed_task.degradation_type,
                visual_evidence=(attempt.diagnosis.visual_evidence if attempt.diagnosis is not None else []),
                route_to=DEGRADATION_TO_EXPERT[parsed_task.degradation_type],
            )
            routing_source = RoutingSource.ORACLE_LABEL

        if actual_diagnosis is None:
            return self._finish_diagnosis_failure(
                rollout.rollout_id,
                routing_mode,
                attempt,
                output_dir,
            )

        routing_payload = {
            "routing_mode": routing_mode.value,
            "routing_source": routing_source.value if routing_source is not None else None,
            "predicted_parse_status": attempt.parse_status.value,
            "actual_diagnosis": actual_diagnosis.model_dump(mode="json"),
        }
        with agl.operation(
            name="routing.select",
            **{"restoration.routing_source": routing_source.value if routing_source is not None else "none"},
        ) as operation:
            operation.set_output(routing_payload)
            agl.emit_object(routing_payload, attributes={"restoration.record_type": "routing_decision"})

        controller = self.factory.build(parsed_task, routed_expert_override=actual_diagnosis.route_to)
        workflow_result = controller.run(
            parsed_task,
            trajectory_id=rollout.rollout_id,
            trace=True,
            diagnosis_override=actual_diagnosis,
        )
        if workflow_result.state.final_reward is None:
            raise RuntimeError("controller completed without a final reward")
        result_path = output_dir / "stage_e_result.json"
        result = StageEWorkflowResult(
            trajectory_id=rollout.rollout_id,
            routing_mode=routing_mode,
            routing_source=routing_source,
            diagnosis_attempt=attempt,
            actual_diagnosis=actual_diagnosis,
            workflow_result=workflow_result,
            termination_reason=workflow_result.state.termination_reason or "unknown",
            final_reward=workflow_result.state.final_reward,
            result_path=str(result_path),
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.results[rollout.rollout_id] = result
        return result.final_reward

    def _finish_diagnosis_failure(
        self,
        trajectory_id: str,
        routing_mode: RoutingMode,
        attempt: VLMDiagnosisAttempt,
        output_dir: Path,
    ) -> float:
        final_reward = -self.config.vlm.diagnosis_failure_penalty
        rejection_payload = {
            "routing_mode": routing_mode.value,
            "routing_source": None,
            "parse_status": attempt.parse_status.value,
            "termination_reason": "diagnosis_failed",
        }
        with agl.operation(name="routing.rejected") as operation:
            operation.set_output(rejection_payload)
            agl.emit_object(rejection_payload, attributes={"restoration.record_type": "routing_decision"})
        result_path = output_dir / "stage_e_result.json"
        result = StageEWorkflowResult(
            trajectory_id=trajectory_id,
            routing_mode=routing_mode,
            routing_source=None,
            diagnosis_attempt=attempt,
            actual_diagnosis=None,
            workflow_result=None,
            termination_reason="diagnosis_failed",
            final_reward=final_reward,
            result_path=str(result_path),
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.results[trajectory_id] = result
        return final_reward


class StageFImageRestorationAgent(agl.LitAgent[dict[str, Any]]):
    """Run stage F with real diagnosis and replay or strict VLM expert decisions."""

    def __init__(
        self,
        config: StageFExampleConfig,
        factory: RealControllerFactory,
        diagnosis_agent: VLMDegradationDiagnosisAgent,
        *,
        expert_client: object | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.factory = factory
        self.diagnosis_agent = diagnosis_agent
        self.expert_client = expert_client
        self.results: dict[str, StageFWorkflowResult] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Resolve diagnosis, select one expert decision source, and run the workflow."""

        del resources
        parsed_task = StageFRestorationTask.model_validate(task)
        routing_mode = parsed_task.routing_mode or self.config.vlm.routing_mode
        expert_mode = parsed_task.expert_decision_mode or self.config.expert_vlm.decision_mode
        output_dir = Path(parsed_task.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with agl.operation(name="diagnosis_agent.vlm_prediction", **{"agent.name": "diagnosis_agent"}) as operation:
            attempt = self.diagnosis_agent.diagnose(parsed_task.image_path)
            attempt_payload = attempt.model_dump(mode="json")
            operation.set_output(attempt_payload)
            agl.emit_object(attempt_payload, attributes={"restoration.record_type": "vlm_diagnosis_attempt"})

        actual_diagnosis: DiagnosisResult | None
        routing_source: RoutingSource | None
        if routing_mode == RoutingMode.PREDICTED_STRICT:
            actual_diagnosis = attempt.diagnosis
            routing_source = RoutingSource.VLM_PREDICTION if actual_diagnosis is not None else None
        else:
            actual_diagnosis = DiagnosisResult(
                primary_type=parsed_task.degradation_type,
                visual_evidence=(attempt.diagnosis.visual_evidence if attempt.diagnosis is not None else []),
                route_to=DEGRADATION_TO_EXPERT[parsed_task.degradation_type],
            )
            routing_source = RoutingSource.ORACLE_LABEL

        if actual_diagnosis is None:
            return self._finish_diagnosis_failure(
                rollout.rollout_id,
                routing_mode,
                expert_mode,
                attempt,
                output_dir,
            )

        routing_payload = {
            "routing_mode": routing_mode.value,
            "routing_source": routing_source.value if routing_source is not None else None,
            "predicted_parse_status": attempt.parse_status.value,
            "actual_diagnosis": actual_diagnosis.model_dump(mode="json"),
            "expert_decision_mode": expert_mode.value,
        }
        with agl.operation(
            name="routing.select",
            **{"restoration.routing_source": routing_source.value if routing_source is not None else "none"},
        ) as operation:
            operation.set_output(routing_payload)
            agl.emit_object(routing_payload, attributes={"restoration.record_type": "routing_decision"})

        experts = self._build_experts(parsed_task, actual_diagnosis.route_to, expert_mode)
        controller = self.factory.build(
            parsed_task,
            routed_expert_override=actual_diagnosis.route_to,
            experts_override=experts,
        )
        workflow_result = controller.run(
            parsed_task,
            trajectory_id=rollout.rollout_id,
            trace=True,
            diagnosis_override=actual_diagnosis,
        )
        if workflow_result.state.final_reward is None:
            raise RuntimeError("controller completed without a final reward")
        result_path = output_dir / "stage_f_result.json"
        result = StageFWorkflowResult(
            trajectory_id=rollout.rollout_id,
            routing_mode=routing_mode,
            routing_source=routing_source,
            expert_decision_mode=expert_mode,
            diagnosis_attempt=attempt,
            actual_diagnosis=actual_diagnosis,
            workflow_result=workflow_result,
            termination_reason=workflow_result.state.termination_reason or "unknown",
            final_reward=workflow_result.state.final_reward,
            result_path=str(result_path),
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.results[rollout.rollout_id] = result
        return result.final_reward

    def _build_experts(
        self,
        task: StageFRestorationTask,
        routed_expert: ExpertName,
        expert_mode: ExpertDecisionMode,
    ) -> dict[ExpertName, ExpertAgent]:
        experts: dict[ExpertName, ExpertAgent] = {
            expert_name: ReplayExpertAgent.from_actions(expert_name, ["stop"], self.factory.tool_registry)
            for expert_name in ExpertName
        }
        if expert_mode == ExpertDecisionMode.REPLAY:
            experts[routed_expert] = ReplayExpertAgent.from_actions(
                routed_expert,
                task.scripted_actions,
                self.factory.tool_registry,
            )
        else:
            experts[routed_expert] = VLMRestorationExpertAgent(
                self.config.expert_vlm,
                routed_expert,
                self.factory.tool_registry,
                max_steps=self.config.workflow.max_steps,
                client=self.expert_client,
            )
        return experts

    def _finish_diagnosis_failure(
        self,
        trajectory_id: str,
        routing_mode: RoutingMode,
        expert_mode: ExpertDecisionMode,
        attempt: VLMDiagnosisAttempt,
        output_dir: Path,
    ) -> float:
        final_reward = -self.config.vlm.diagnosis_failure_penalty
        rejection_payload = {
            "routing_mode": routing_mode.value,
            "routing_source": None,
            "expert_decision_mode": expert_mode.value,
            "parse_status": attempt.parse_status.value,
            "termination_reason": "diagnosis_failed",
        }
        with agl.operation(name="routing.rejected") as operation:
            operation.set_output(rejection_payload)
            agl.emit_object(rejection_payload, attributes={"restoration.record_type": "routing_decision"})
        result_path = output_dir / "stage_f_result.json"
        result = StageFWorkflowResult(
            trajectory_id=trajectory_id,
            routing_mode=routing_mode,
            routing_source=None,
            expert_decision_mode=expert_mode,
            diagnosis_attempt=attempt,
            actual_diagnosis=None,
            workflow_result=None,
            termination_reason="diagnosis_failed",
            final_reward=final_reward,
            result_path=str(result_path),
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.results[trajectory_id] = result
        return final_reward
