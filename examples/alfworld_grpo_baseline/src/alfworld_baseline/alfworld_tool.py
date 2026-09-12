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
        # This cache is intentionally opt-in.  The Qwen3.5-2B profile enables
        # it in its private tool config; other model profiles retain the old
        # close-per-trajectory behavior.  A cached object is never shared by
        # two active instances and is reloaded with the requested game file on
        # every acquire.
        self._env_pool_size = max(0, int(self.config.get("env_pool_size", 0)))
        self._idle_envs: list[Any] = []

    @staticmethod
    def _load_game_without_rebuilding(env: Any, game_file: str) -> tuple[Any, Any]:
        """Load and reset a one-game TextWorld batch without recreating it.

        ``TextworldBatchGymEnv.reset`` closes and recreates its underlying
        environment before loading the next game.  For a pooled ALFWorld
        instance this is unnecessary: the existing ``SyncBatchEnv`` can load a
        new game directly and then reset its existing wrapper chain.
        """
        batch_env = getattr(env, "batch_env", None)
        if batch_env is None or not hasattr(batch_env, "load"):
            return env.reset()
        batch_env.load([str(game_file)])
        env.last_commands = [None]
        env.obs, info = batch_env.reset()
        return env.obs, info

    def _build_env(self, game_file: str) -> Any:
        """Construct one fresh ALFWorld TextWorld environment."""
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        import yaml

        data_root = Path(os.environ.get("ALFWORLD_DATA", ""))
        if not data_root.is_dir():
            raise FileNotFoundError("ALFWORLD_DATA is not configured")
        with (data_root / "base_config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
        alfred = AlfredTWEnv(config, train_eval="train")
        alfred.game_files, alfred.num_games = [str(game_file)], 1
        return alfred.init_env(batch_size=1)

    async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> tuple[str, Any]:
        create_kwargs = kwargs.get("create_kwargs", {})
        game_file = create_kwargs.get("game_file")
        if not game_file:
            raise ValueError("ALFWorldTool.create requires create_kwargs.game_file")
        game_file = str(game_file)
        reused = bool(self._idle_envs)
        env = self._idle_envs.pop() if reused else self._build_env(game_file)
        try:
            # Keep the legacy fresh-instance path unchanged for profiles that
            # do not opt into pooling.  The direct load path is only needed
            # when reusing an already initialized TextWorld batch.
            if reused or self._env_pool_size > 0:
                observation, info = self._load_game_without_rebuilding(env, game_file)
            else:
                observation, info = env.reset()
        except Exception:
            # A poisoned/closed pooled instance must not take down the rollout
            # batch.  Replace it with a fresh environment once.
            try:
                env.close()
            except Exception:
                pass
            env = self._build_env(game_file)
            observation, info = self._load_game_without_rebuilding(env, game_file)
        instance = instance_id or str(uuid4())
        self._instances[instance] = {"env": env, "observation": observation[0], "info": info, "steps": 0}
        return instance, ToolResponse()

    def get_state(self, instance_id: str) -> tuple[str, tuple[str, ...]]:
        """Return the authoritative current observation and action list.

        The agent loop uses this immediately after ``create`` so the first
        model prompt is built from the same environment instance that will
        execute the first action, rather than trusting a stale parquet prompt.
        """
        state = self._instances[instance_id]
        info = state["info"]
        return str(state["observation"]), tuple(str(a) for a in info["admissible_commands"][0])

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[Any, float, dict]:
        state = self._instances[instance_id]
        action = parameters.get("action")
        info = state["info"]
        actions = tuple(info["admissible_commands"][0])
        registry = ALFWorldToolRegistry(actions)
        try:
            action = registry.validate_action(action)
        except ValueError:
            current_observation = str(state["observation"])
            current_actions = tuple(str(a) for a in info["admissible_commands"][0])
            text = (
                "Tool execution failed: invalid action.\n\n"
                f"The action {action!r} is not admissible in the current state.\n\n"
                f"Current observation:\n{current_observation}\n\n"
                "Current admissible actions (copy exactly one):\n"
                + "\n".join(current_actions)
            )
            return ToolResponse(text=text), 0.0, {
                "error": "invalid_action",
                "action": action,
                "observation": current_observation,
                "admissible_commands": current_actions,
            }
        (observations,), (rewards,), (done,), next_info = state["env"].step([action])
        state["observation"], state["info"] = observations, next_info
        state["steps"] += 1
        won = bool(next_info.get("won", [False])[0])
        truncated = state["steps"] >= int(self.config.get("max_steps", 50)) and not won
        done = bool(done or truncated)
        reward = float(rewards)
        text = f"Observation:\n{observations}\n\nAdmissible actions:\n{chr(10).join(next_info['admissible_commands'][0])}"
        return ToolResponse(text=text), reward, {
            "action": action,
            "observation": str(observations),
            "won": won,
            "done": done,
            "truncated": truncated,
            "admissible_commands": next_info["admissible_commands"][0],
        }

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        state = self._instances.pop(instance_id, None)
        if state is not None:
            env = state["env"]
            if len(self._idle_envs) < self._env_pool_size:
                # Do not close a pooled object.  The next acquire reloads its
                # game file and resets the wrapper state before use.
                self._idle_envs.append(env)
            else:
                env.close()
