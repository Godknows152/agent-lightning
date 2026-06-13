"""Real no-reference IQA evaluator backed by an isolated `verl` process."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import cast

from config import EvaluatorSettings, IQAMetricConfig
from schemas import EvaluationResult, ExecutionStatus
from subprocess_utils import parse_result_json, resolve_python_command, validate_image_file


def _normalize_score(score: float, metric: IQAMetricConfig) -> float:
    normalized = (score - metric.minimum) / (metric.maximum - metric.minimum)
    normalized = min(1.0, max(0.0, normalized))
    return normalized if metric.higher_is_better else 1.0 - normalized


class PyiqaSubprocessEvaluator:
    """Compute, normalize, and aggregate configured IQA metrics."""

    def __init__(self, settings: EvaluatorSettings, *, improvement_epsilon: float) -> None:
        self.settings = settings
        self.improvement_epsilon = improvement_epsilon
        self.python_command = resolve_python_command(settings.environment_name, settings.python_executable)
        self.entrypoint = Path(settings.entrypoint).expanduser().resolve()
        self.iqa_repo = Path(settings.iqa_repo).expanduser().resolve()

    def evaluate(
        self,
        image_path: str,
        *,
        previous_score: float | None,
        original_score: float | None,
        best_score: float | None,
    ) -> EvaluationResult:
        """Evaluate one readable image and derive trajectory-relative deltas."""

        started = time.perf_counter()
        fallback_score = previous_score if previous_score is not None else 0.0
        try:
            image = Path(image_path).expanduser().resolve()
            validate_image_file(image)
            if not self.entrypoint.is_file():
                raise FileNotFoundError(f"IQA entrypoint does not exist: {self.entrypoint}")
            if not self.iqa_repo.is_dir():
                raise FileNotFoundError(f"IQA-PyTorch repository does not exist: {self.iqa_repo}")
            metric_names = [metric.name for metric in self.settings.metrics]
            command = [
                *self.python_command,
                str(self.entrypoint),
                "--repo",
                str(self.iqa_repo),
                "--metrics",
                ",".join(metric_names),
                "--input",
                str(image),
                "--device",
                self.settings.device,
            ]
            environment = os.environ.copy()
            environment.setdefault("HF_HUB_OFFLINE", "1")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
                env=environment,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"IQA subprocess exited with {completed.returncode}: {detail[-2000:]}")
            process_result = parse_result_json(completed.stdout)
            if process_result.get("status") != "success":
                raise RuntimeError(f"IQA subprocess reported failure: {process_result}")
            raw_payload = process_result.get("raw_scores")
            if not isinstance(raw_payload, dict):
                raise ValueError("IQA subprocess result is missing raw_scores")
            typed_raw_payload = cast(dict[str, object], raw_payload)
            raw_scores: dict[str, float] = {}
            for name in metric_names:
                value = typed_raw_payload.get(name)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"IQA metric {name} did not return a numeric score")
                raw_scores[name] = float(value)
            normalized_scores = {
                metric.name: _normalize_score(raw_scores[metric.name], metric) for metric in self.settings.metrics
            }
            total_weight = sum(metric.weight for metric in self.settings.metrics)
            aggregate = (
                sum(normalized_scores[metric.name] * metric.weight for metric in self.settings.metrics) / total_weight
            )
            previous = aggregate if previous_score is None else previous_score
            original = aggregate if original_score is None else original_score
            best = aggregate if best_score is None else best_score
            delta_previous = aggregate - previous
            delta_original = aggregate - original
            delta_best = aggregate - best
            is_new_best = best_score is None or aggregate > best + self.improvement_epsilon
            direction = "improved" if delta_previous > self.improvement_epsilon else "did not improve"
            return EvaluationResult(
                status=ExecutionStatus.SUCCESS,
                raw_scores=raw_scores,
                normalized_scores=normalized_scores,
                aggregate_score=aggregate,
                delta_from_previous=delta_previous,
                delta_from_original=delta_original,
                delta_from_best=delta_best,
                is_new_best=is_new_best,
                feedback=f"IQA aggregate {direction}; aggregate_score={aggregate:.4f}.",
                error=None,
                latency_seconds=time.perf_counter() - started,
                metadata={
                    **process_result,
                    "environment_name": self.settings.environment_name,
                    "device": self.settings.device,
                    "normalization": {
                        metric.name: {
                            "minimum": metric.minimum,
                            "maximum": metric.maximum,
                            "higher_is_better": metric.higher_is_better,
                            "weight": metric.weight,
                        }
                        for metric in self.settings.metrics
                    },
                },
            )
        except Exception as error:
            return EvaluationResult(
                status=ExecutionStatus.FAILED,
                raw_scores={},
                normalized_scores={},
                aggregate_score=fallback_score,
                delta_from_previous=0.0,
                delta_from_original=0.0,
                delta_from_best=0.0,
                is_new_best=False,
                feedback="IQA evaluation failed; the candidate image was rejected.",
                error=f"{type(error).__name__}: {error}",
                latency_seconds=time.perf_counter() - started,
                metadata={
                    "environment_name": self.settings.environment_name,
                    "device": self.settings.device,
                },
            )
