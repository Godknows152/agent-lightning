"""Display complete ALFWorld episodes using native veRL's validation dump hook."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def trajectory_samples(
    keys: list[str], inputs: list[str], outputs: list[str], scores: list[float]
) -> list[tuple[str, str, float]]:
    """Group native ``{uid}_{session}_{step}`` rows into chronological transcripts.

    Preserve the exact decoded inputs and responses, including malformed model
    outputs. The score belongs to the episode, so take it from its final row.
    """
    sessions: dict[str, list[tuple[int, str, str, float]]] = defaultdict(list)
    for key, prompt, response, score in zip(keys, inputs, outputs, scores, strict=True):
        parts = key.rsplit("_", 2)
        session, index = (f"{parts[0]}_{parts[1]}", int(parts[2])) if len(parts) == 3 else (key, 0)
        sessions[session].append((index, prompt, response, score))

    samples = []
    for steps in sessions.values():
        steps.sort(key=lambda step: step[0])
        transcript = [f"Trajectory: {len(steps)} decision steps"]
        for index, prompt, response, _ in steps:
            transcript.append(f"===== Step {index + 1} =====\n[Input]\n{prompt}\n\n[Model response]\n{response}")
        samples.append((steps[0][1], "\n\n".join(transcript), steps[-1][3]))
    return samples


class ALFWorldValidationLoggingMixin:
    """Replace final-step previews with episode previews, leaving native metrics intact.

    Native V1 calls the preview hook with final rows before calling the dump hook
    with every row. Defer the preview until the full data is available. A configured
    ``validation_data_dir`` enables this path and also retains the per-step JSONL.
    """

    def _validate(self) -> dict[str, float]:
        self._alfworld_full_validation = bool(self.config.trainer.get("validation_data_dir"))
        try:
            return super()._validate()
        finally:
            self._alfworld_full_validation = False

    def _maybe_log_val_generations(self, inputs: list[str], outputs: list[str], scores: list[float]) -> None:
        if not getattr(self, "_alfworld_full_validation", False):
            super()._maybe_log_val_generations(inputs=inputs, outputs=outputs, scores=scores)

    def _dump_generations(
        self,
        inputs: list[str],
        outputs: list[str],
        gts: list[Any],
        scores: list[float],
        reward_extra_infos_dict: dict[str, list[Any]],
        dump_path: str,
    ) -> None:
        super()._dump_generations(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
        )
        if not getattr(self, "_alfworld_full_validation", False) or not self.config.trainer.log_val_generations:
            return
        samples = trajectory_samples(reward_extra_infos_dict["uid"], inputs, outputs, scores)
        # Reuse native sample selection and logger integration, including its
        # configured sample count and deterministic shuffling.
        super()._maybe_log_val_generations(
            inputs=[sample[0] for sample in samples],
            outputs=[sample[1] for sample in samples],
            scores=[sample[2] for sample in samples],
        )
