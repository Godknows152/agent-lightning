"""old-VERL BaseTool adapter for one ALFWorld text trajectory."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .tool_registry import ALFWorldToolRegistry, TOOL_NAME

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
except ImportError:  # Allows parser/validator tests in the lightweight alfworld env.
    BaseTool = object  # type: ignore[misc,assignment]
    OpenAIFunctionToolSchema = Any  # type: ignore[misc,assignment]
    ToolResponse = Any  # type: ignore[misc,assignment]


class ALFWorldTool(BaseTool):
    """Execute validated text commands against an isolated AlfredTWEnv instance."""

    def __init__(self, config: dict[str, Any], tool_schema: Any = None):
        if tool_schema is not None:
            super().__init__(config, tool_schema)
        self.config = config
        self.tool_schema = tool_schema
        self.name = TOOL_NAME
        self._instances: dict[str, Any] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> tuple[str, Any]:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        import yaml

        create_kwargs = kwargs.get("create_kwargs", {})
        game_file = create_kwargs.get("game_file")
        if not game_file:
            raise ValueError("ALFWorldTool.create requires create_kwargs.game_file")
        data_root = Path(os.environ.get("ALFWORLD_DATA", ""))
        if not data_root.is_dir():
            raise FileNotFoundError("ALFWORLD_DATA is not configured")
        with (data_root / "base_config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
        env = AlfredTWEnv(config, train_eval="train")
        env.game_files, env.num_games = [str(game_file)], 1
        env = env.init_env(batch_size=1)
        instance = instance_id or str(uuid4())
        observation, info = env.reset()
        self._instances[instance] = {"env": env, "observation": observation[0], "info": info, "steps": 0}
        return instance, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[Any, float, dict]:
        state = self._instances[instance_id]
        action = parameters.get("action")
        info = state["info"]
        actions = tuple(info["admissible_commands"][0])
        registry = ALFWorldToolRegistry(actions)
        try:
            action = registry.validate_action(action)
        except ValueError as exc:
            # Protocol penalty is emitted once for this assistant turn.  It is
            # deliberately separate from the native TextWorld reward.
            penalty = float(
                self.config.get("invalid_action_penalty", self.config.get("format_penalty", -0.05))
            )
            return ToolResponse(text=f"Invalid ALFWorld action: {exc}"), penalty, {"error": "invalid_action", "action": action}
        (observations,), (rewards,), (done,), next_info = state["env"].step([action])
        state["observation"], state["info"] = observations, next_info
        state["steps"] += 1
        truncated = state["steps"] >= int(self.config.get("max_steps", 50)) and not done
        done = bool(done or truncated)
        reward = float(rewards)
        text = f"Observation:\n{observations}\n\nAdmissible actions:\n{chr(10).join(next_info['admissible_commands'][0])}"
        return ToolResponse(text=text), reward, {"action": action, "won": bool(next_info.get("won", [False])[0]), "done": done, "truncated": truncated, "admissible_commands": next_info["admissible_commands"][0]}

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        state = self._instances.pop(instance_id, None)
        if state is not None:
            state["env"].close()
