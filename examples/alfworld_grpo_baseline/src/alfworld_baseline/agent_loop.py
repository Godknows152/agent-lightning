"""ALFWorld step samples for the unmodified native veRL V1 trainer."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids

from .budget import ALFWorldDecisionBudget
from .prompt_profiles import get_prompt_profile
from .prompts_qwen35 import NONTHINKING_PROMPT_VERSION, PROMPT_VERSION, QWEN35_ALFWORLD_CHAT_TEMPLATE
from .tool_registry import ALFWorldToolRegistry
from .thinking import ThinkingToolParser, tool_output
from .text_actions import parse_text_action
from .xml_actions import parse_xml_decision


@register("alfworld_tool_agent")
class ALFWorldToolAgentLoop(ToolAgentLoop):
    """Rebuild each decision context and emit one native training row per step."""

    TOOL_CALL_START_TOKEN = "<tool_call>"
    TOOL_CALL_END_TOKEN = "</tool_call>"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config = getattr(self, "config", {})
        variables = config.get("variables", {}) if hasattr(config, "get") else {}
        prompt_name = variables.get("PROMPT_PROFILE", "gigpo")
        self._prompt_profile = get_prompt_profile(str(prompt_name))
        self._gigpo_prompt = self._prompt_profile.PROMPT_VERSION.startswith("alfworld_gigpo_")
        model_profile = str(variables.get("MODEL_PROFILE", os.environ.get("ALFWORLD_MODEL_PROFILE", ""))).lower()
        self._is_qwen35_alfworld = model_profile.startswith("qwen35") or self.processor is not None
        self._text_actions = False
        self._xml_actions = self._gigpo_prompt or self._is_qwen35_alfworld
        if self._xml_actions:
            self.apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
            self.apply_chat_template_kwargs["chat_template"] = (
                self._prompt_profile.QWEN3_ALFWORLD_CHAT_TEMPLATE if self._gigpo_prompt
                else QWEN35_ALFWORLD_CHAT_TEMPLATE
            )
            # Honor the composed training config instead of forcing thinking.
            self.apply_chat_template_kwargs.setdefault("enable_thinking", self._gigpo_prompt)
        self._thinking_enabled = bool(
            self.apply_chat_template_kwargs.get("enable_thinking", self._gigpo_prompt)
        )
        if self._thinking_enabled and not self._text_actions:
            self.tool_parser = ThinkingToolParser(self.tool_parser, self.tokenizer)
        tool = self.tools.get("alfworld_action")
        tool_config = getattr(tool, "config", {}) or {}
        self._environment_budget = ALFWorldDecisionBudget.from_tool_config(tool_config)
        if self._environment_budget is None:
            raise ValueError("Native ALFWorld requires environment_driven=true")
        if self.response_length < self._environment_budget.max_new_tokens_per_turn:
            raise ValueError("response_length must cover max_new_tokens_per_turn")

    async def apply_chat_template(self, messages, *, tools=None, images=None, videos=None):
        # Text-only ALFWorld uses the exact same template for every fresh state.
        options = dict(getattr(self, "apply_chat_template_kwargs", {}))
        return normalize_token_ids(self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True, **options
        ))

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> list[AgentLoopOutput]:
        if kwargs.get("data_source", "alfworld") != "alfworld":
            raise ValueError("ALFWorld loop requires data_source=alfworld")
        data = AgentData(
            messages=list(kwargs["raw_prompt"]), image_data=None, video_data=None,
            audio_data=None, mm_processor_kwargs={}, metrics={}, request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        data.data_source = "alfworld"
        data._active_tools = self.tools
        data._active_tool_schemas = self.tool_schemas
        data.total_tool_calls = 0
        data.action_history = []
        data.successful_action_history = []
        data.step_outputs = []
        try:
            state = await self._handle_pending_state(data, sampling_params)
            while state != AgentState.TERMINATED:
                if state == AgentState.GENERATING:
                    state = await self._generate_environment_decision(data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._process_environment_decision(data)
                else:
                    raise RuntimeError(f"Unexpected ALFWorld state: {state}")
            return self._finalize_step_outputs(data)
        finally:
            await self._release_native_tool_instances(data)

    def _finalize_step_outputs(self, data: AgentData) -> list[AgentLoopOutput]:
        """Keep the original episode reward and final-row GRPO contract."""
        from .reward import compute_score

        extra = dict(data.extra_fields, tool_rewards=list(data.tool_rewards))
        score = compute_score("alfworld", extra_info=extra)
        # Native V1 normalizes final rows once per session, then broadcasts
        # the resulting advantage to all steps. Every row keeps its actual
        # inference prompt and original generated IDs/log probabilities.
        for index, output in enumerate(data.step_outputs):
            output.reward_score = score
            output.extra_fields.update(extra)
            output.extra_fields.update(
                alfworld_step_index=index,
                alfworld_is_final_step=index == len(data.step_outputs) - 1,
                reward_extra_info={"score": score},
            )
            if index == len(data.step_outputs) - 1:
                output.metrics = type(output.metrics)(**data.metrics)
        return data.step_outputs

    def _selected_prompt_profile(self):
        """Return the configured profile, including for lightweight test doubles."""
        profile = getattr(self, "_prompt_profile", None)
        if profile is not None:
            return profile
        return get_prompt_profile("gigpo" if getattr(self, "_gigpo_prompt", False) else "qwen35_v7")

    async def _get_or_create_tool_instance(
        self,
        tool_name: str,
        tool: Any,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> str:
        """Keep one ALFWorld environment alive for the entire episode."""
        instances = getattr(agent_data, "_alfworld_tool_instances", None)
        if instances is None:
            instances = {}
            agent_data._alfworld_tool_instances = instances
        instance_id = instances.get(tool_name)
        if instance_id is None:
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            instances[tool_name] = instance_id
        return instance_id

    async def _release_native_tool_instances(self, agent_data: AgentData) -> None:
        """Release episode instances, including on generation errors."""
        instances = getattr(agent_data, "_alfworld_tool_instances", {})
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        for tool_name, instance_id in list(instances.items()):
            tool = active_tools.get(tool_name)
            if tool is None:
                continue
            try:
                await tool.release(instance_id)
            finally:
                instances.pop(tool_name, None)

    @staticmethod
    def _mission_from_messages(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            match = re.search(
                r"(?:ALFWorld task|Task goal \(not an executable action\)|Task):\s*(.+?)(?:\n\n|$)",
                content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
            match = re.search(r"Your task is to:\s*(.+?)(?:\n\n|$)", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return "Complete the ALFWorld task."

    @staticmethod
    def _mission_from_observation(observation: str) -> str | None:
        match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation)
        return match.group(1).strip() if match else None

    @staticmethod
    def _compact_observation(observation: str, mission: str) -> str:
        # ALFWorld repeats the task in the initial observation. Keep the outer
        # task field as the single task instruction and remove that duplicate.
        kept_lines = []
        for line in observation.splitlines():
            if line.strip().startswith("Your task is"):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _state_prompt_messages(
        self,
        *,
        mission: str,
        observation: str,
        actions: tuple[str, ...],
        history: tuple[str, ...] = (),
    ) -> list[dict[str, str]]:
        """Build the next request from current state and optional decision history."""

        return [
            {
                "role": "user",
                "content": self._selected_prompt_profile().build_user_prompt(
                    mission=mission,
                    observation=observation,
                    admissible_actions=actions,
                    history=history,
                    enable_thinking=getattr(self, "_thinking_enabled", True),
                    avoid_repeated_actions=bool(
                        getattr(self, "apply_chat_template_kwargs", {}).get("avoid_repeated_actions", False)
                    ),
                ),
            }
        ]

    async def _set_authoritative_initial_prompt(self, agent_data: AgentData) -> None:
        """Create the tool first and build turn zero from its actual state."""
        active_tools = getattr(agent_data, "_active_tools", None) or getattr(self, "tools", {})
        tool = active_tools.get("alfworld_action")
        if tool is None or not hasattr(tool, "get_state"):
            return
        instance_id = await self._get_or_create_tool_instance(
            "alfworld_action", tool, agent_data.tools_kwargs, agent_data
        )
        observation, actions = tool.get_state(instance_id)
        mission = self._mission_from_observation(observation) or self._mission_from_messages(agent_data.messages)
        agent_data.alfworld_mission = mission
        agent_data.alfworld_recent_history = []
        agent_data.alfworld_decision_history = []
        agent_data.alfworld_prompt_history = []
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # Remove the dataset's possibly stale state and any custom system
        # message. The selected profile supplies the canonical instructions.
        agent_data.messages = self._state_prompt_messages(
            mission=mission, observation=observation, actions=actions
        )
        # The XML template renders a compact static schema, without action enums.
        agent_data._active_tool_schemas = (
            [] if getattr(self, "_text_actions", False) else [ALFWorldToolRegistry(actions).build_tool_schema()]
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        if getattr(agent_data, "data_source", "") == "alfworld":
            await self._set_authoritative_initial_prompt(agent_data)
            if getattr(self, "_text_actions", False) or getattr(self, "_xml_actions", False):
                agent_data.extra_fields["alfworld_prompt_version"] = (
                    (self._selected_prompt_profile().PROMPT_VERSION if getattr(self, "_thinking_enabled", True)
                     else self._selected_prompt_profile().NONTHINKING_PROMPT_VERSION)
                    if getattr(self, "_gigpo_prompt", False)
                    else PROMPT_VERSION if getattr(self, "_thinking_enabled", True) else NONTHINKING_PROMPT_VERSION
                )
            agent_data.extra_fields.setdefault("alfworld_no_tool_call_penalty_count", 0)
            agent_data.extra_fields["alfworld_no_action_category"] = ""
            for category in ("thinking_unclosed", "tool_call_format", "overlong_thinking", "other"):
                agent_data.extra_fields[f"alfworld_no_action_{category}_count"] = 0
            agent_data.extra_fields.setdefault("alfworld_invalid_tool_call_penalty_count", 0)
            agent_data.extra_fields.setdefault("alfworld_valid_tool_call_count", 0)
            agent_data.extra_fields.setdefault("alfworld_repeated_action_penalty_count", 0)
            agent_data.alfworld_last_valid_action = None
            agent_data.alfworld_action_streak_length = 0
            if getattr(self, "_environment_budget", None) is not None:
                agent_data.extra_fields.update({
                    "alfworld_decision_steps": 0,
                    "alfworld_terminal_reason": "running",
                    "alfworld_environment_finished": False,
                })
        agent_data.prompt_ids = await self.apply_chat_template(
            agent_data.messages, tools=getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        )
        agent_data.generation_prompt_ids = list(agent_data.prompt_ids)
        return AgentState.GENERATING

    async def _rebuild_generation_prompt_after_tool(self, agent_data: AgentData) -> None:
        """Build the next native sample from current state and recent history."""
        if getattr(agent_data, "data_source", "") != "alfworld":
            return
        if agent_data.extra_fields.get("alfworld_environment_finished"):
            return
        tool_metrics = getattr(agent_data, "alfworld_last_tool_metrics", {}) or {}
        observation = str(tool_metrics.get("observation", agent_data.alfworld_current_observation))
        actions = tuple(
            str(a) for a in tool_metrics.get("admissible_commands", agent_data.alfworld_current_actions)
        )
        # Store chronological action/status history separately from observations.
        # Do not replay previous reasoning or obsolete action lists in the prompt.
        agent_data.alfworld_recent_history = []
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # The XML template renders a compact static schema, without action enums.
        agent_data._active_tool_schemas = (
            [] if getattr(self, "_text_actions", False) else [ALFWorldToolRegistry(actions).build_tool_schema()]
        )
        messages = self._state_prompt_messages(
            mission=agent_data.alfworld_mission,
            observation=observation,
            actions=actions,
            history=tuple(getattr(agent_data, "alfworld_prompt_history", ()))
            if getattr(self, "_gigpo_prompt", False)
            else tuple(getattr(agent_data, "alfworld_decision_history", ()))
            if (getattr(self, "_text_actions", False) or getattr(self, "_xml_actions", False)) else (),
        )
        agent_data.messages = messages
        agent_data.generation_prompt_ids = await self.apply_chat_template(
            messages,
            tools=getattr(agent_data, "_active_tool_schemas", self.tool_schemas),
            images=None,
            videos=None,
        )

    ALFWORLD_NO_TOOL_CALL_PENALTY = -2.0
    ALFWORLD_INVALID_TOOL_CALL_PENALTY = -2.0
    ALFWORLD_REPEATED_ACTION_PENALTY = -0.1

    def _record_alfworld_penalty(
        self,
        agent_data: AgentData,
        kind: Literal["no_tool_call", "invalid_tool_call", "repeated_action"],
        *,
        append_reward: bool,
        prior_occurrences: int = 0,
    ) -> float:
        """Record one of three mutually exclusive decision penalties.

        Legacy internal counter names are retained. In v7 no_tool_call means
        absent/incomplete post-thinking XML; invalid_tool_call means a complete
        but schema-invalid decision or an action rejected by the environment.
        Each missing action costs -2 and consumes one decision without ending the trajectory.
        Repeated valid actions are counted across the whole trajectory. The
        aggregate repeated-action penalty is applied by the final reward
        function, where successful trajectories can be exempted. Protocol
        failures receive their own immediate penalties.
        """
        if kind == "no_tool_call":
            count_key = "alfworld_no_tool_call_penalty_count"
            value = self.ALFWORLD_NO_TOOL_CALL_PENALTY
        elif kind == "invalid_tool_call":
            count_key = "alfworld_invalid_tool_call_penalty_count"
            value = self.ALFWORLD_INVALID_TOOL_CALL_PENALTY
        elif kind == "repeated_action":
            if prior_occurrences < 1:
                raise ValueError("Repeated actions require at least one preceding valid occurrence in the streak")
            count_key = "alfworld_repeated_action_penalty_count"
            value = self.ALFWORLD_REPEATED_ACTION_PENALTY
        else:  # pragma: no cover - Literal callers should make this unreachable.
            raise ValueError(f"unknown ALFWorld penalty kind: {kind!r}")

        if kind != "repeated_action":
            agent_data.alfworld_last_valid_action = None
            agent_data.alfworld_action_streak_length = 0
        extra_fields = getattr(agent_data, "extra_fields", None)
        if extra_fields is None:
            extra_fields = {}
            agent_data.extra_fields = extra_fields
        extra_fields[count_key] = int(extra_fields.get(count_key, 0) or 0) + 1
        if append_reward:
            agent_data.tool_rewards.append(value)
        return value

    def _invalid_tool_call_penalty(self, agent_data: AgentData) -> float:
        """Count an invalid parsed call and return its synthetic reward."""
        return self._record_alfworld_penalty(agent_data, "invalid_tool_call", append_reward=False)

    def _has_complete_tool_call_schema(self, token_ids: list[int]) -> bool:
        """Check the structural envelope before trusting parser output.

        Qwen XML parsing intentionally has a back-off path for partially
        generated calls. Such a fragment can still become a ``FunctionCall``
        with empty arguments, but it is category 1 (unparseable/incomplete),
        not category 2. Semantic argument errors are left for ``_call_tool``.
        """
        text, _ = tool_output(
            self.tokenizer.decode(token_ids, skip_special_tokens=False),
            enable_thinking=getattr(self, "_thinking_enabled", False),
        )
        start_token = self.TOOL_CALL_START_TOKEN
        end_token = self.TOOL_CALL_END_TOKEN
        start = text.find(start_token)
        if start < 0:
            return False
        end = text.find(end_token, start + len(start_token))
        if end < 0:
            return False

        payload = text[start + len(start_token) : end].strip()
        if payload.startswith("<function="):
            if payload.count("<function=") != payload.count("</function>"):
                return False
            if payload.count("<parameter=") != payload.count("</parameter>"):
                return False
        return True

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        return await self._generate_environment_decision(agent_data, sampling_params)

    def _apply_tool_call_format_guardrails(
        self, agent_data: AgentData, token_ids: list[int], log_probs: list[float] | None
    ) -> tuple[list[int], list[float] | None]:
        """Preserve every generated token; classify the entire decision."""
        return token_ids, log_probs

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[Any, float, dict]:
        """Execute a parsed ALFWorld call and classify execution failures.

        Parsing failures are handled before this method and belong to the
        no-tool-call category. Any parsed call that cannot be executed is
        category 2, including unknown tool names, invalid JSON/parameter
        schemas, invalid actions, and tool exceptions.
        """
        if getattr(agent_data, "data_source", "") == "alfworld":
            # Failed parsing must not replay the previous turn's feedback.
            agent_data.alfworld_last_tool_metrics = {"action": "unknown", "error": "invalid_tool_call"}
        try:
            decoded_arguments = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError):
            penalty = self._invalid_tool_call_penalty(agent_data)
            return (
                ToolResponse(text=f"Error when executing tool: invalid JSON arguments for '{tool_call.name}'"),
                penalty,
                {"error": "invalid_json_arguments", "skip_tool_call_reward": True},
            )

        if (
            not isinstance(decoded_arguments, dict)
            or set(decoded_arguments) != {"action"}
            or not isinstance(decoded_arguments.get("action"), str)
        ):
            penalty = self._invalid_tool_call_penalty(agent_data)
            return (
                ToolResponse(text=f"Error when executing tool: invalid arguments schema for '{tool_call.name}'"),
                penalty,
                {"error": "invalid_arguments_schema", "skip_tool_call_reward": True},
            )

        active_tools = getattr(agent_data, "_active_tools", None) or getattr(self, "tools", {})
        try:
            tool = active_tools[tool_call.name]
            instance_id = await self._get_or_create_tool_instance(tool_call.name, tool, tools_kwargs, agent_data)
            response, reward, metrics = await tool.execute(instance_id, decoded_arguments, agent_data=agent_data)
        except KeyError:
            penalty = self._invalid_tool_call_penalty(agent_data)
            return (
                ToolResponse(text=f"Error when executing tool: unknown tool '{tool_call.name}'"),
                penalty,
                {"error": "unknown_tool", "requested_tool": tool_call.name, "skip_tool_call_reward": True},
            )
        except Exception as exc:
            penalty = self._invalid_tool_call_penalty(agent_data)
            return (
                ToolResponse(text=f"Error when executing tool: {exc}"),
                penalty,
                {"error": type(exc).__name__, "skip_tool_call_reward": True},
            )
        if isinstance(metrics, dict) and metrics.get("error"):
            # The parser produced a call, but the environment rejected it
            # (for example an inadmissible action). This is category 2, and
            # the returned value is appended by the processing phase.
            reward = self._invalid_tool_call_penalty(agent_data)
        elif getattr(agent_data, "data_source", "") == "alfworld" and isinstance(metrics, dict):
            # Repeated-action penalties are intentionally deferred until the
            # trajectory reward is computed. This lets reward.py apply the
            # success gate and avoids subtracting the penalty from each turn.
            action = metrics.get("action")
            if isinstance(action, str) and action:
                agent_data.alfworld_last_valid_action = action
                agent_data.alfworld_action_streak_length = 1
            else:
                agent_data.alfworld_last_valid_action = None
                agent_data.alfworld_action_streak_length = 0
        elif getattr(agent_data, "data_source", "") == "alfworld":
            agent_data.alfworld_last_valid_action = None
            agent_data.alfworld_action_streak_length = 0
        if getattr(agent_data, "data_source", "") == "alfworld" and isinstance(metrics, dict):
            agent_data.alfworld_last_tool_metrics = dict(metrics)
            if metrics.get("admissible_commands"):
                agent_data.alfworld_current_actions = tuple(str(a) for a in metrics["admissible_commands"])
            if metrics.get("observation") is not None:
                agent_data.alfworld_current_observation = str(metrics["observation"])
            if bool(metrics.get("done")) or bool(metrics.get("truncated")):
                agent_data.extra_fields["alfworld_environment_finished"] = True
                if metrics.get("won"):
                    reason = "success"
                elif metrics.get("truncated"):
                    reason = "environment_timeout"
                else:
                    reason = "env_failure"
                agent_data.extra_fields["alfworld_terminal_reason"] = reason
        return response, reward, metrics

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        return await self._process_environment_decision(agent_data)

    async def _generate_environment_decision(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Generate one native prompt/response row, bounded per decision."""
        agent_data.extra_fields["alfworld_no_action_category"] = ""
        budget = self._environment_budget
        assert budget is not None
        prompt = list(agent_data.generation_prompt_ids or agent_data.prompt_ids)
        if len(prompt) > getattr(self, "prompt_length", float("inf")):
            raise ValueError(f"ALFWorld step prompt has {len(prompt)} tokens, exceeding prompt_length")
        generation_params = dict(sampling_params)
        generation_params.pop("max_tokens", None)
        generation_params.pop("max_generated_response_length", None)
        generation_params["max_new_tokens"] = budget.max_new_tokens_per_turn
        # SGLang may skip tokenizer initialization and use a model-config EOS
        # different from the tokenizer EOS used by SFT (Qwen3.5: <|im_end|>).
        # Send it on every decision, including greedy validation requests.
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids = set(generation_params.get("stop_token_ids") or [])
            stop_token_ids.add(self.tokenizer.eos_token_id)
            generation_params["stop_token_ids"] = sorted(stop_token_ids)
        generation_params["ignore_eos"] = False
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=prompt,
                sampling_params=generation_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # Fail loudly on backend contract violations instead of silently
        # clipping training tokens or accepting a trajectory with no loss slots.
        if not output.token_ids or len(output.token_ids) > budget.max_new_tokens_per_turn:
            raise RuntimeError("ALFWorld generation must return 1..max_new_tokens_per_turn tokens")
        if sampling_params.get("logprobs") and output.log_probs is None:
            raise RuntimeError("ALFWorld requested rollout log probabilities but the server returned none")
        if output.log_probs is not None and len(output.log_probs) != len(output.token_ids):
            raise RuntimeError("ALFWorld generation returned misaligned token log probabilities")
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output.routed_experts is not None:
            raise NotImplementedError("Environment-driven ALFWorld currently supports dense models only")

        agent_data.assistant_turns += 1
        response_ids, log_probs = self._apply_tool_call_format_guardrails(
            agent_data, output.token_ids, output.log_probs
        )
        if len(response_ids) > self.response_length:
            raise RuntimeError("ALFWorld response row too small; refusing to truncate")
        agent_data.response_ids = list(response_ids)
        agent_data.response_mask = [1] * len(response_ids)
        agent_data.response_logprobs = list(log_probs) if log_probs is not None else []
        if not hasattr(agent_data, "step_outputs"):
            agent_data.step_outputs = []
        agent_data.step_outputs.append(AgentLoopOutput(
            prompt_ids=prompt, response_ids=list(response_ids),
            response_mask=[1] * len(response_ids), response_logprobs=log_probs,
            num_turns=1, metrics={}, extra_fields=dict(getattr(output, "extra_fields", None) or {}),
        ))

        if getattr(self, "_text_actions", False):
            # Legacy test adapter only. All configured ALFWorld profiles set
            # _text_actions=False and use the Qwen3 XML branch below.
            text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            action = parse_text_action(text, enable_thinking=self._thinking_enabled)
            agent_data.alfworld_pending_text_action = action
            # Internal adapter only: no serialized tool call enters training tokens.
            agent_data.tool_calls = [] if action is None else [FunctionCall(
                name="alfworld_action", arguments=json.dumps({"action": action})
            )]
            return AgentState.PROCESSING_TOOLS

        if getattr(self, "_xml_actions", False):
            decision = parse_xml_decision(
                self.tokenizer.decode(response_ids, skip_special_tokens=False),
                enable_thinking=self._thinking_enabled,
            )
            agent_data.alfworld_xml_decision = decision
            agent_data.alfworld_pending_action = decision.action
            agent_data.tool_calls = [] if decision.status != "valid" else [FunctionCall(
                name="alfworld_action", arguments=json.dumps({"action": decision.action})
            )]
            return AgentState.PROCESSING_TOOLS

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        typed_schemas = [OpenAIFunctionToolSchema.model_validate(schema) for schema in schemas]
        try:
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(response_ids, typed_schemas)
        except Exception:
            # Parser-level failures are indistinguishable from an incomplete or
            # malformed schema at the task level: category 1, not category 2.
            agent_data.tool_calls = []
        if agent_data.tool_calls and not self._has_complete_tool_call_schema(response_ids):
            agent_data.tool_calls = []
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        # A missing or malformed call is penalized in the tool phase, preserving coordinator balance.
        agent_data.alfworld_last_tool_metrics = {"error": "no_tool_call"}
        # Still pass through the tool phase: shared phase coordinators require
        # one after_tool arrival for every nonterminal generation, even a no-op.
        return AgentState.PROCESSING_TOOLS

    async def _process_environment_decision(self, agent_data: AgentData) -> AgentState:
        """Execute at most one call and retain rewards, not tool-text loss slots."""
        # _call_tool updates the current observation; history needs the state
        # seen by the model before it chose this action (GiGPO SimpleMemory).
        agent_data.alfworld_previous_observation = agent_data.alfworld_current_observation
        decision = getattr(agent_data, "alfworld_xml_decision", None)
        if getattr(self, "_xml_actions", False) and decision is not None:
            agent_data.extra_fields["alfworld_last_decision_reason"] = decision.reason
            if decision.status == "no_action":
                # Exactly one no-action detail is recorded. Overlong thinking
                # takes precedence over the broader unclosed-thinking bucket.
                overlong = (
                    decision.reason == "unclosed_thinking"
                    and len(agent_data.response_ids) >= self._environment_budget.max_new_tokens_per_turn
                )
                agent_data.extra_fields["alfworld_no_action_category"] = (
                    "overlong_thinking" if overlong
                    else "thinking_unclosed" if decision.reason == "unclosed_thinking"
                    else "tool_call_format" if decision.reason in {
                        "missing_or_incomplete_call", "malformed_xml", "incomplete_parameter"
                    }
                    else "other"
                )
            if decision.status == "invalid_action":
                self._record_alfworld_penalty(agent_data, "invalid_tool_call", append_reward=True)
                agent_data.alfworld_last_tool_metrics = {"error": decision.reason}
                return await self._finish_environment_decision(agent_data)
        if not agent_data.tool_calls:
            self._record_alfworld_penalty(agent_data, "no_tool_call", append_reward=True)
            agent_data.alfworld_last_tool_metrics = {"error": "no_tool_call"}
            category = agent_data.extra_fields.get("alfworld_no_action_category") or "other"
            count_key = f"alfworld_no_action_{category}_count"
            agent_data.extra_fields[count_key] = int(agent_data.extra_fields.get(count_key, 0)) + 1
            return await self._finish_environment_decision(agent_data)
        tool_call = agent_data.tool_calls[0]
        agent_data.total_tool_calls += 1
        with simple_timer("tool_calls", agent_data.metrics):
            _, reward, metrics = await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        metrics = metrics or {}
        agent_data.alfworld_last_tool_metrics = dict(metrics)
        if getattr(self, "_xml_actions", False):
            agent_data.extra_fields["alfworld_last_decision_reason"] = metrics.get("error") or "executed"
        if not metrics.get("error"):
            # This increment is deliberately in the environment-driven
            # decision path: _finish_environment_decision increments
            # alfworld_decision_steps for the same step immediately after it.
            extra_fields = agent_data.extra_fields
            extra_fields["alfworld_valid_tool_call_count"] = int(
                extra_fields.get("alfworld_valid_tool_call_count", 0) or 0
            ) + 1
        if reward is not None:
            # The tool normalizes a successful terminal transition to +10.
            # Keep this guard for test doubles and alternate tool adapters.
            if metrics.get("won") and getattr(agent_data, "data_source", "") == "alfworld":
                reward = 10.0
            agent_data.tool_rewards.append(float(reward))
        action = metrics.get("action")
        if action:
            agent_data.action_history.append(action)
            if not metrics.get("error"):
                # Count every occurrence after the first one, regardless of
                # whether it is consecutive. The final reward function applies
                # escalating costs (-0.1, -0.15, ...) for unsuccessful trajectories only.
                prior_occurrences = agent_data.successful_action_history.count(action)
                if prior_occurrences > 0:
                    extra_fields = agent_data.extra_fields
                    extra_fields["alfworld_repeated_action_penalty_count"] = int(
                        extra_fields.get("alfworld_repeated_action_penalty_count", 0) or 0
                    ) + 1
                agent_data.successful_action_history.append(action)
        agent_data.tool_calls = []
        return await self._finish_environment_decision(agent_data)

    async def _finish_environment_decision(self, agent_data: AgentData) -> AgentState:
        """Share the tool's Max Steps budget across valid and failed decisions.

        Missing actions and parsed invalid actions both cost -2. Both leave
        TextWorld unchanged, consume one decision step, and allow another attempt.
        Environment completion takes precedence on the last allowed step,
        so a last-step success is not labelled truncated.
        """
        if getattr(self, "_text_actions", False) or getattr(self, "_xml_actions", False):
            history = getattr(agent_data, "alfworld_decision_history", [])
            action_field = "alfworld_pending_action" if getattr(self, "_xml_actions", False) else "alfworld_pending_text_action"
            action = getattr(agent_data, action_field, None)
            metrics = getattr(agent_data, "alfworld_last_tool_metrics", {}) or {}
            error = metrics.get("error")
            decision = getattr(agent_data, "alfworld_xml_decision", None)
            rejected = error and (action is not None or (decision is not None and decision.status == "invalid_action"))
            status = "rejected" if rejected else ("no action" if action is None else "executed")
            history.append(f"{action or ('(invalid XML call)' if rejected else '(no parseable action)')} [{status}]")
            agent_data.alfworld_decision_history = history
            if getattr(self, "_gigpo_prompt", False):
                prompt_history = getattr(agent_data, "alfworld_prompt_history", [])
                step = len(prompt_history) + 1
                observation = agent_data.alfworld_previous_observation
                prompt_history.append(f"[Observation {step}: '{observation}', Action {step}: '{action}']")
                agent_data.alfworld_prompt_history = prompt_history
        budget = self._environment_budget
        assert budget is not None
        steps = int(agent_data.extra_fields.get("alfworld_decision_steps", 0)) + 1
        agent_data.extra_fields["alfworld_decision_steps"] = steps
        agent_data.user_turns += 1
        if agent_data.extra_fields.get("alfworld_environment_finished"):
            if agent_data.extra_fields.get("alfworld_terminal_reason") == "truncated":
                agent_data.extra_fields["alfworld_terminal_reason"] = "environment_timeout"
            await self._release_native_tool_instances(agent_data)
            return AgentState.TERMINATED
        if steps >= budget.max_steps:
            agent_data.extra_fields["alfworld_terminal_reason"] = "decision_limit"
            await self._release_native_tool_instances(agent_data)
            return AgentState.TERMINATED
        await self._rebuild_generation_prompt_after_tool(agent_data)
        return AgentState.GENERATING
