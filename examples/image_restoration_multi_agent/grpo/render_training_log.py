#!/usr/bin/env python3
"""Render Ray/VERL output as compact logs plus stable local progress bars."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RAY_PREFIX_PATTERN = re.compile(r"^\((?P<actor>[^)]+)\)\s*")
TASK_PROGRESS_PATTERN = re.compile(r"Completed\s+(?P<current>\d+)/(?P<total>\d+)\s+tasks")
TRAINING_PROGRESS_PATTERN = re.compile(r"Training Progress:\s*.*?\|\s*(?P<current>\d+)/(?P<total>\d+)\s*\[")
TOOL_PARSER_MARKERS = (
    "qwen3_coder_tool_parser.py",
    "qwen3coder_tool_parser.py",
    "tool_parser.py",
)
TOOL_PARSER_ERROR_MARKER = "Error in extracting tool call from response"


@dataclass(frozen=True)
class ProgressEvent:
    """One parsed progress update."""

    kind: str
    current: int
    total: int


class LogLineParser:
    """Normalize Ray output and collapse expected Qwen3 parser tracebacks."""

    def parse(self, raw_line: str) -> tuple[ProgressEvent | None, str | None]:
        line = ANSI_PATTERN.sub("", raw_line).strip("\r\n")
        if not line:
            return None, None

        task_match = TASK_PROGRESS_PATTERN.search(line)
        if task_match:
            return (
                ProgressEvent(
                    kind="rollout",
                    current=int(task_match.group("current")),
                    total=int(task_match.group("total")),
                ),
                None,
            )

        training_match = TRAINING_PROGRESS_PATTERN.search(line)
        if training_match:
            return (
                ProgressEvent(
                    kind="training",
                    current=int(training_match.group("current")),
                    total=int(training_match.group("total")),
                ),
                None,
            )

        if any(marker in line for marker in TOOL_PARSER_MARKERS):
            if TOOL_PARSER_ERROR_MARKER in line:
                return None, "[WARN] Qwen3 tool call parse failed; response marked invalid_tool_call."
            return None, None

        prefix_match = RAY_PREFIX_PATTERN.match(line)
        if prefix_match:
            actor = prefix_match.group("actor").split(" pid=", maxsplit=1)[0]
            line = f"[{actor}] {line[prefix_match.end():]}"
        return None, line


class ProgressBars:
    """Maintain one training bar and one per-batch rollout bar."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.training: tqdm[object] | None = None
        self.rollout: tqdm[object] | None = None

    def update(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        if event.kind == "training":
            self.training = self._update_bar(
                self.training,
                event,
                description="Training",
                position=0,
            )
        else:
            self.rollout = self._update_bar(
                self.rollout,
                event,
                description="Rollout tasks",
                position=1,
            )

    @staticmethod
    def _update_bar(
        bar: tqdm[object] | None,
        event: ProgressEvent,
        *,
        description: str,
        position: int,
    ) -> tqdm[object]:
        should_reset = bar is None or bar.total != event.total or event.current < bar.n
        if should_reset:
            if bar is not None:
                bar.close()
            bar = tqdm(
                total=event.total,
                desc=description,
                position=position,
                leave=True,
                dynamic_ncols=True,
            )
        bar.n = min(event.current, event.total)
        bar.refresh()
        return bar

    def write(self, line: str) -> None:
        if self.enabled:
            tqdm.write(line)
        else:
            print(line, flush=True)

    def close(self) -> None:
        if self.rollout is not None:
            self.rollout.close()
        if self.training is not None:
            self.training.close()


def _write_progress_snapshot(log_file: TextIO, event: ProgressEvent) -> None:
    label = "Training progress" if event.kind == "training" else "Rollout tasks"
    log_file.write(f"{label}: {event.current}/{event.total}\n")
    log_file.flush()


def render_stream(source: TextIO, log_file: TextIO, *, use_progress_bars: bool) -> None:
    """Render one combined stdout/stderr stream."""

    parser = LogLineParser()
    bars = ProgressBars(enabled=use_progress_bars)
    last_progress: dict[str, tuple[int, int]] = {}
    try:
        for raw_line in source:
            event, line = parser.parse(raw_line)
            if event is not None:
                bars.update(event)
                progress = (event.current, event.total)
                if last_progress.get(event.kind) != progress:
                    _write_progress_snapshot(log_file, event)
                    last_progress[event.kind] = progress
                continue
            if line is None:
                continue
            log_file.write(line + "\n")
            log_file.flush()
            bars.write(line)
    finally:
        bars.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=sys.stderr.isatty(),
        help="Render interactive progress bars; defaults to terminal detection.",
    )
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        render_stream(sys.stdin, log_file, use_progress_bars=args.progress)


if __name__ == "__main__":
    main()
