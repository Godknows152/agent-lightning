"""Tests for compact GRPO log rendering."""

from __future__ import annotations

import io
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/image_restoration_multi_agent"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from grpo.render_training_log import LogLineParser, ProgressEvent, render_stream  # noqa: E402


def test_parser_extracts_task_and_training_progress() -> None:
    parser = LogLineParser()

    task_event, task_line = parser.parse("\x1b[36m(TaskRunner pid=1)\x1b[0m Completed 32/256 tasks...\n")
    training_event, training_line = parser.parse("\rTraining Progress:  25%|██▌       | 3/12 [01:00<03:00, 20.0s/it]\r")

    assert task_event == ProgressEvent(kind="rollout", current=32, total=256)
    assert task_line is None
    assert training_event == ProgressEvent(kind="training", current=3, total=12)
    assert training_line is None


def test_parser_compacts_qwen3_traceback_to_one_line() -> None:
    parser = LogLineParser()
    error_line = (
        "\x1b[36m(vLLMHttpServer pid=2)\x1b[0m ERROR "
        "[qwen3_coder_tool_parser.py:148] Error in extracting tool call from response.\n"
    )
    traceback_line = (
        "\x1b[36m(vLLMHttpServer pid=2)\x1b[0m ERROR "
        "[qwen3_coder_tool_parser.py:148] ValueError: incomplete function tag\n"
    )

    assert parser.parse(error_line) == (
        None,
        "[WARN] Qwen3 tool call parse failed; response marked invalid_tool_call.",
    )
    assert parser.parse(traceback_line) == (None, None)


def test_render_stream_writes_compact_log_without_progress_spam() -> None:
    source = io.StringIO(
        "(TaskRunner pid=1) Completed 8/256 tasks...\n"
        "(TaskRunner pid=1) Completed 8/256 tasks...\n"
        "(TaskRunner pid=1) Completed 16/256 tasks...\n"
        "(TaskRunner pid=1) useful message\n"
    )
    destination = io.StringIO()

    render_stream(source, destination, use_progress_bars=False)

    assert destination.getvalue() == (
        "Rollout tasks: 8/256\n" "Rollout tasks: 16/256\n" "[TaskRunner] useful message\n"
    )
