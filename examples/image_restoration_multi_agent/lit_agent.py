"""Agent Lightning wrapper for the deterministic restoration controller."""

from __future__ import annotations

from typing import Any

from factory import DeterministicControllerFactory
from schemas import RestorationTask, WorkflowResult

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
