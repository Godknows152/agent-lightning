"""Deterministic IQA evaluator for workflow and trace tests."""

from __future__ import annotations

from schemas import EvaluationResult, ExecutionStatus


class ScriptedEvaluator:
    """Consume a task-provided score sequence one image at a time."""

    def __init__(
        self,
        scores: list[float],
        *,
        improvement_epsilon: float,
        fail_indices: set[int] | None = None,
    ) -> None:
        self.scores = list(scores)
        self.improvement_epsilon = improvement_epsilon
        self.fail_indices = fail_indices or set()
        self.call_count = 0

    def evaluate(
        self,
        image_path: str,
        *,
        previous_score: float | None,
        original_score: float | None,
        best_score: float | None,
    ) -> EvaluationResult:
        """Return the next deterministic score and its deltas."""

        del image_path
        evaluation_index = self.call_count
        self.call_count += 1
        fallback_score = previous_score if previous_score is not None else 0.0
        if evaluation_index in self.fail_indices:
            return EvaluationResult(
                status=ExecutionStatus.FAILED,
                raw_scores={"mock_quality": fallback_score},
                normalized_scores={"mock_quality": fallback_score},
                aggregate_score=fallback_score,
                delta_from_previous=0.0,
                delta_from_original=0.0,
                delta_from_best=0.0,
                is_new_best=False,
                feedback="Scripted evaluation failed.",
                error=f"scripted evaluation failure at index {evaluation_index}",
            )

        score = self.scores[evaluation_index] if evaluation_index < len(self.scores) else self.scores[-1]
        previous = score if previous_score is None else previous_score
        original = score if original_score is None else original_score
        best = score if best_score is None else best_score
        is_new_best = best_score is None or score > best + self.improvement_epsilon
        delta_previous = score - previous
        delta_original = score - original
        delta_best = score - best
        direction = "improved" if delta_previous > self.improvement_epsilon else "did not improve"
        return EvaluationResult(
            status=ExecutionStatus.SUCCESS,
            raw_scores={"mock_quality": score},
            normalized_scores={"mock_quality": score},
            aggregate_score=score,
            delta_from_previous=delta_previous,
            delta_from_original=delta_original,
            delta_from_best=delta_best,
            is_new_best=is_new_best,
            feedback=f"Mock quality {direction}; aggregate_score={score:.4f}.",
            error=None,
        )
