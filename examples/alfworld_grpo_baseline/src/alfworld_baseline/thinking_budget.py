"""Experimental SGLang-only thinking budget for v5 token-in/token-out sampling.

Not enabled in PPO: forcing the closing token changes its sampling logprob to
zero. A training rollout must account for that deterministic transition in its
loss mask/replay before enabling this processor with bypass old logprobs.
"""
from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor


class Qwen35TokenIdsOnlyBudget(ThinkingBudgetLogitProcessor):
    """Diagnostic control: fixes IDs but retains upstream whole-prompt scanning."""

    THINKING_START_TOKEN_ID = 248068
    THINKING_END_TOKEN_ID = 248069
    NEW_LINE_TOKEN_ID = 198


class ALFWorldThinkingBudgetLogitProcessor(Qwen35TokenIdsOnlyBudget):
    """Cap generated thinking tokens after the final open thinking prefix.

    The system instructions may themselves include both thinking delimiters.
    Only the last open prefix is active. Budget excludes the prompt's newline
    and the closing token; force exactly one closing token at the boundary.
    Early natural closure is allowed. This is a hard cap, not a minimum length.
    """

    FORCE_NEWLINE_BEFORE_CLOSE = False

    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list or []):
            if not params:
                continue
            budget = params.get('thinking_budget')
            if type(budget) is not int or budget < 0:
                continue
            req = params.get('__req__')
            if req is None:
                continue
            prompt = req.origin_input_ids
            starts = [j for j, token in enumerate(prompt) if token == self.THINKING_START_TOKEN_ID]
            if not starts or self.THINKING_END_TOKEN_ID in prompt[starts[-1] + 1:]:
                continue
            output = req.output_ids
            if self.THINKING_END_TOKEN_ID in output or len(output) < budget:
                continue
            logits[i, :] = -float('inf')
            target = self.THINKING_END_TOKEN_ID
            if self.FORCE_NEWLINE_BEFORE_CLOSE and (not output or output[-1] != self.NEW_LINE_TOKEN_ID):
                target = self.NEW_LINE_TOKEN_ID
            logits[i, target] = 0.0
        return logits


class ALFWorldNewlineThinkingBudgetLogitProcessor(ALFWorldThinkingBudgetLogitProcessor):
    """Upstream-style termination: force newline then closing tag at budget.

    The delimiter prefix can add one newline token beyond the thinking budget.
    This control tests whether native-style whitespace improves action output.
    """

    FORCE_NEWLINE_BEFORE_CLOSE = True
