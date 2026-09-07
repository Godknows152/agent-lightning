"""ALFWorld-only old-VERL AgentLoop extension."""
from __future__ import annotations

import json
import copy
import os
import re
from typing import Any, Literal

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.profiler import simple_timer

from .budget import ALFWorldDecisionBudget
from .prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
from .tool_registry import ALFWorldToolRegistry
from .thinking import ThinkingToolParser, tool_output


@register("alfworld_tool_agent")
class ALFWorldToolAgentLoop(ToolAgentLoop):
    """ALFWorld loop with authoritative state and optional full conversation history.

    History mode retains all assistant/feedback pairs with one static protocol
    schema. Per-turn inference contexts are replayed for policy training.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Replace the stock Qwen3.5 template only for Qwen3.5 ALFWorld runs.
        # The stock template contains a generic example_function_name and
        # normal-answer branch that compete with the dynamic schema. Qwen2.5
        # keeps its existing Hermes path unchanged.
        model_profile = os.environ.get("ALFWORLD_MODEL_PROFILE", "").lower()
        self._is_qwen35_alfworld = model_profile.startswith("qwen35") or self.processor is not None
        if self._is_qwen35_alfworld:
            self.apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
            self.apply_chat_template_kwargs["chat_template"] = QWEN35_ALFWORLD_CHAT_TEMPLATE
        self._thinking_enabled = self._is_qwen35_alfworld and bool(
            self.apply_chat_template_kwargs.get("enable_thinking", False)
        )
        if self._thinking_enabled:
            self.tool_parser = ThinkingToolParser(self.tool_parser, self.tokenizer)
        tool = self.tools.get("alfworld_action")
        tool_config = getattr(tool, "config", {}) or {}
        self._environment_budget = ALFWorldDecisionBudget.from_tool_config(tool_config)
        self._history_context = bool(tool_config.get("history_context", False))
        self._history_max_turns = int(tool_config.get("history_max_turns", 4))
        if self._history_max_turns <= 0:
            raise ValueError("ALFWorld history_max_turns must be positive")
        if self._environment_budget is not None and self.response_length < self._environment_budget.response_capacity:
            raise ValueError(
                "ALFWorld response storage is too small for max_steps * max_new_tokens_per_turn. "
                "Launch via alfworld_baseline.main_ppo to derive capacity before creating workers."
            )

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
    ) -> list[dict[str, str]]:
        """Build the next request from the goal and latest state only."""

        return [
            {
                "role": "user",
                "content": build_user_prompt(
                    mission=mission,
                    observation=observation,
                    admissible_actions=actions,
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
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # Remove the dataset's possibly stale state and any custom system
        # message. The chat template supplies the canonical tool instructions.
        agent_data.messages = self._state_prompt_messages(
            mission=mission, observation=observation, actions=actions
        )
        # The action enum is part of the model-facing tool schema, not merely
        # prose in the prompt. Rebuild it from the authoritative environment
        # state. Execution still validates membership (no constrained decoder).
        agent_data._active_tool_schemas = [ALFWorldToolRegistry(actions).build_tool_schema()]
        if getattr(self, "_history_context", False):
            # A persistent protocol schema must not freeze the initial state's enum.
            schema = copy.deepcopy(agent_data._active_tool_schemas[0])
            action = schema["function"]["parameters"]["properties"]["action"]
            action.pop("enum", None)
            action["description"] = "Copy exactly one action from the latest Current admissible actions list."
            agent_data.alfworld_protocol_schemas = [schema]
            agent_data._active_tool_schemas = [schema]


    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        if getattr(agent_data, "data_source", "") == "alfworld":
            await self._set_authoritative_initial_prompt(agent_data)
            agent_data.extra_fields.setdefault("alfworld_no_tool_call_penalty_count", 0)
            agent_data.extra_fields.setdefault("alfworld_invalid_tool_call_penalty_count", 0)
            agent_data.extra_fields.setdefault("alfworld_valid_tool_call_count", 0)
            if getattr(self, "_environment_budget", None) is not None:
                agent_data.extra_fields.update({
                    "alfworld_decision_steps": 0,
                    "alfworld_terminal_reason": "running",
                    "alfworld_environment_finished": False,
                })
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _rebuild_generation_prompt_after_tool(self, agent_data: AgentData) -> None:
        """Refresh state and build a history-aware or current-state-only request."""
        if getattr(agent_data, "data_source", "") != "alfworld":
            return
        if agent_data.extra_fields.get("alfworld_environment_finished"):
            return
        tool_metrics = getattr(agent_data, "alfworld_last_tool_metrics", {}) or {}
        observation = str(tool_metrics.get("observation", agent_data.alfworld_current_observation))
        actions = tuple(
            str(a) for a in tool_metrics.get("admissible_commands", agent_data.alfworld_current_actions)
        )
        # Keep this compatibility field empty: history is retained internally
        # only for accounting/debugging and is never part of model context.
        agent_data.alfworld_recent_history = []
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # Refresh the enum after every environment transition. The next model
        # request and the tool parser must use the same latest action list.
        agent_data._active_tool_schemas = [ALFWorldToolRegistry(actions).build_tool_schema()]
        messages = self._state_prompt_messages(
            mission=agent_data.alfworld_mission,
            observation=observation,
            actions=actions,
        )
        # Only the next inference context is compacted. prompt_ids retains the
        # complete VERL trajectory and its response mask for training. Per-turn
        # contexts are replayed by FSDP for actor/ref logprobs and gradients.
        if getattr(self, "_history_context", False):
            # Keep the protocol-bearing initial user message and only the most
            # recent complete assistant/feedback pairs.  The generated
            # trajectory remains complete in prompt_ids; this bound controls
            # the replay inputs and prevents quadratic context growth.
            feedback = messages[0]["content"].split("\n\nChoose exactly one next action", 1)[0]
            agent_data.messages.append({"role": "tool", "content": feedback})
            history = agent_data.messages
            pair_count = (len(history) - 1) // 2
            history_max_turns = getattr(self, "_history_max_turns", 4)
            if pair_count > history_max_turns:
                first = history[:1]
                recent = history[-(history_max_turns * 2) :]
                agent_data.messages = first + recent
            messages = agent_data.messages
            prompt_schemas = getattr(
                agent_data, "alfworld_protocol_schemas", getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            )
        else:
            agent_data.messages = messages
            prompt_schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        agent_data.generation_prompt_ids = await self.apply_chat_template(
            messages,
            tools=prompt_schemas,
            images=None,
            videos=None,
        )

    ALFWORLD_NO_TOOL_CALL_PENALTY = -0.1
    ALFWORLD_INVALID_TOOL_CALL_PENALTY = -0.1

    def _record_alfworld_penalty(
        self,
        agent_data: AgentData,
        kind: Literal["no_tool_call", "invalid_tool_call"],
        *,
        append_reward: bool,
    ) -> float:
        """Record exactly one ALFWorld protocol penalty.

        ALFWorld deliberately keeps its protocol accounting separate from the
        restoration loop's generic ``penalty_records`` mechanism. There are
        only two mutually exclusive categories: no parsed tool call and a
        parsed but invalid tool call. ``append_reward`` is false when the
        caller returns the penalty to VERL, whose processing phase appends the
        returned value to ``tool_rewards`` itself.
        """
        if kind == "no_tool_call":
            count_key = "alfworld_no_tool_call_penalty_count"
            value = self.ALFWORLD_NO_TOOL_CALL_PENALTY
        elif kind == "invalid_tool_call":
            count_key = "alfworld_invalid_tool_call_penalty_count"
            value = self.ALFWORLD_INVALID_TOOL_CALL_PENALTY
        else:  # pragma: no cover - Literal callers should make this unreachable.
            raise ValueError(f"unknown ALFWorld penalty kind: {kind!r}")

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
        """Use per-decision generation for opted-in ALFWorld runs; preserve legacy behavior."""
        if getattr(self, "_environment_budget", None) is not None and agent_data.data_source == "alfworld":
            return await self._generate_environment_decision(agent_data, sampling_params)
        is_alfworld = getattr(agent_data, "data_source", "") == "alfworld"
        if is_alfworld:
            prompt = list(agent_data.generation_prompt_ids or agent_data.prompt_ids)
            offset = len(agent_data.response_mask)
            previous_turns = agent_data.assistant_turns
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if is_alfworld and agent_data.assistant_turns > previous_turns:
            agent_data.extra_fields.setdefault("alfworld_turn_contexts", []).append({
                "prompt_ids": prompt,
                "response_ids": list(agent_data.response_ids),
                "response_offset": offset,
            })
            # The legacy loop terminates immediately when extraction yields no
            # call. Classify that turn after extraction so malformed XML, an
            # incomplete schema, and ordinary text all share category 1.
            if agent_data.tool_calls and not self._has_complete_tool_call_schema(agent_data.response_ids):
                agent_data.tool_calls = []
            if not agent_data.tool_calls:
                self._record_alfworld_penalty(agent_data, "no_tool_call", append_reward=True)
        if getattr(agent_data, "data_source", "") != "alfworld" or agent_data.tool_calls:
            return state
        return state

    def _apply_tool_call_format_guardrails(
        self, agent_data: AgentData, token_ids: list[int], log_probs: list[float] | None
    ) -> tuple[list[int], list[float] | None]:
        """Trim after the first complete call without altering ALFWorld reward."""
        if getattr(agent_data, "data_source", "") != "alfworld" or not token_ids:
            return super()._apply_tool_call_format_guardrails(agent_data, token_ids, log_probs)

        format_token_ids = self._strip_trailing_termination_tokens(token_ids)
        full_text = self.tokenizer.decode(format_token_ids)
        text, reasoning_offset = tool_output(
            full_text, enable_thinking=getattr(self, "_thinking_enabled", False)
        )
        start = text.find(self.TOOL_CALL_START_TOKEN)
        end = text.find(self.TOOL_CALL_END_TOKEN, start) if start >= 0 else -1
        if start < 0 or end < 0:
            return token_ids, log_probs

        call_end = end + len(self.TOOL_CALL_END_TOKEN)
        if getattr(self, "_thinking_enabled", False):
            # Keep the original reasoning + call IDs and aligned logprobs.
            # XML examples inside reasoning must not truncate the real call.
            target = reasoning_offset + call_end
            keep_token_count = next(
                (i for i in range(1, len(token_ids) + 1)
                 if len(self.tokenizer.decode(token_ids[:i])) >= target),
                None,
            )
        else:
            keep_token_count = self._first_complete_tool_call_token_count(token_ids)
        if keep_token_count is None or keep_token_count >= len(token_ids):
            return token_ids, log_probs
        trimmed_log_probs = log_probs[:keep_token_count] if log_probs else None
        return token_ids[:keep_token_count], trimmed_log_probs

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
        if getattr(agent_data, "data_source", "") == "alfworld" and isinstance(metrics, dict):
            agent_data.alfworld_last_tool_metrics = dict(metrics)
            if metrics.get("admissible_commands"):
                agent_data.alfworld_current_actions = tuple(str(a) for a in metrics["admissible_commands"])
            if metrics.get("observation") is not None:
                agent_data.alfworld_current_observation = str(metrics["observation"])
            if bool(metrics.get("done")) or bool(metrics.get("truncated")):
                agent_data.extra_fields["alfworld_environment_finished"] = True
                agent_data.extra_fields["alfworld_terminal_reason"] = (
                    "truncated" if metrics.get("truncated") else "done"
                )
        return response, reward, metrics

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        if getattr(self, "_environment_budget", None) is not None and agent_data.data_source == "alfworld":
            return await self._process_environment_decision(agent_data)
        state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "data_source", "") == "alfworld" and agent_data.extra_fields.get(
            "alfworld_environment_finished"
        ):
            return AgentState.TERMINATED
        return state

    async def _generate_environment_decision(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Generate one bounded decision without trajectory-token/turn cutoffs.

        Only model tokens occupy loss slots. Observations live in the recorded
        real inference prompts and are replayed by the ALFWorld FSDP path.
        This keeps storage provably bounded by max_steps * per-turn tokens,
        rather than dropping tool feedback or truncating the final decision.
        """
        budget = self._environment_budget
        assert budget is not None
        prompt = list(agent_data.generation_prompt_ids or agent_data.prompt_ids)
        generation_params = dict(sampling_params)
        generation_params.pop("max_tokens", None)
        generation_params.pop("max_generated_response_length", None)
        generation_params["max_new_tokens"] = budget.max_new_tokens_per_turn
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
        offset = len(agent_data.response_mask)
        if offset + len(response_ids) > self.response_length:
            raise RuntimeError("ALFWorld response storage invariant violated; refusing to truncate trajectory")
        agent_data.response_ids = list(response_ids)
        if getattr(self, "_history_context", False):
            agent_data.messages.append({
                "role": "assistant",
                "content": self.tokenizer.decode(response_ids, skip_special_tokens=False),
            })
        agent_data.prompt_ids.extend(response_ids)
        agent_data.response_mask.extend([1] * len(response_ids))
        if log_probs is not None:
            agent_data.response_logprobs.extend(log_probs)
        agent_data.extra_fields.setdefault("alfworld_turn_contexts", []).append({
            "prompt_ids": prompt,
            "response_ids": list(response_ids),
            "response_offset": offset,
        })

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

        # A missing or malformed call is a failed decision, not a terminal state.
        agent_data.alfworld_last_tool_metrics = {"error": "no_tool_call"}
        # Still pass through the tool phase: shared phase coordinators require
        # one after_tool arrival for every nonterminal generation, even a no-op.
        return AgentState.PROCESSING_TOOLS

    async def _process_environment_decision(self, agent_data: AgentData) -> AgentState:
        """Execute at most one call and retain rewards, not tool-text loss slots."""
        if not agent_data.tool_calls:
            self._record_alfworld_penalty(agent_data, "no_tool_call", append_reward=True)
            agent_data.alfworld_last_tool_metrics = {"error": "no_tool_call"}
            return await self._finish_environment_decision(agent_data)
        tool_call = agent_data.tool_calls[0]
        agent_data.total_tool_calls += 1
        with simple_timer("tool_calls", agent_data.metrics):
            _, reward, metrics = await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        metrics = metrics or {}
        agent_data.alfworld_last_tool_metrics = dict(metrics)
        if not metrics.get("error"):
            # This increment is deliberately in the environment-driven
            # decision path: _finish_environment_decision increments
            # alfworld_decision_steps for the same step immediately after it.
            extra_fields = agent_data.extra_fields
            extra_fields["alfworld_valid_tool_call_count"] = int(
                extra_fields.get("alfworld_valid_tool_call_count", 0) or 0
            ) + 1
        if reward is not None:
            agent_data.tool_rewards.append(float(reward))
        action = metrics.get("action")
        if action:
            agent_data.action_history.append(action)
            if not metrics.get("error"):
                agent_data.successful_action_history.append(action)
        agent_data.tool_calls = []
        return await self._finish_environment_decision(agent_data)

    async def _finish_environment_decision(self, agent_data: AgentData) -> AgentState:
        """Share the tool's Max Steps budget across valid and failed decisions.

        Invalid output leaves TextWorld state unchanged, contributes no reward,
        and consumes one decision step. It cannot create an unbounded retry
        loop. Environment completion takes precedence on the last allowed step,
        so a last-step success is not labelled truncated.
        """
        budget = self._environment_budget
        assert budget is not None
        steps = int(agent_data.extra_fields.get("alfworld_decision_steps", 0)) + 1
        agent_data.extra_fields["alfworld_decision_steps"] = steps
        agent_data.user_turns += 1
        if agent_data.extra_fields.get("alfworld_environment_finished"):
            if agent_data.extra_fields.get("alfworld_terminal_reason") == "truncated":
                agent_data.extra_fields["alfworld_terminal_reason"] = "max_steps"
            return AgentState.TERMINATED
        if steps >= budget.max_steps:
            agent_data.extra_fields["alfworld_terminal_reason"] = "max_steps"
            return AgentState.TERMINATED
        await self._rebuild_generation_prompt_after_tool(agent_data)
        return AgentState.GENERATING
