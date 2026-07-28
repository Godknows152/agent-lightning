# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from verl.trainer.ppo.ray_trainer import _compute_validation_turn_metrics, _select_validation_turn_counts


def test_validation_num_turns_prefer_actual_tool_call_counts() -> None:
    counts, are_tool_calls = _select_validation_turn_counts(
        {
            "__num_turns__": np.array([7, 11], dtype=np.int32),
            "tool_call_counts": np.array([1, 3], dtype=object),
        }
    )

    assert are_tool_calls is True
    assert counts is not None
    np.testing.assert_array_equal(counts, np.array([1, 3], dtype=np.int64))

    metrics = _compute_validation_turn_metrics([counts], sample_turns_are_tool_calls=are_tool_calls)

    assert metrics["val-aux/num_turns/min"] == 1
    assert metrics["val-aux/num_turns/max"] == 3
    assert metrics["val-aux/num_turns/mean"] == 2.0
    assert metrics["val-aux/tool_call_counts/min"] == 1
    assert metrics["val-aux/tool_call_counts/max"] == 3
    assert metrics["val-aux/tool_call_counts/mean"] == 2.0


def test_validation_num_turns_fall_back_to_chat_turns_without_tool_counts() -> None:
    counts, are_tool_calls = _select_validation_turn_counts(
        {"__num_turns__": np.array([2, 6], dtype=np.int32)}
    )

    assert are_tool_calls is False
    assert counts is not None
    np.testing.assert_array_equal(counts, np.array([2, 6], dtype=np.int64))

    metrics = _compute_validation_turn_metrics([counts], sample_turns_are_tool_calls=are_tool_calls)

    assert metrics["val-aux/num_turns/min"] == 2
    assert metrics["val-aux/num_turns/max"] == 6
    assert metrics["val-aux/num_turns/mean"] == 4.0
    assert "val-aux/tool_call_counts/min" not in metrics


def test_validation_num_turns_are_absent_without_supported_count_fields() -> None:
    counts, are_tool_calls = _select_validation_turn_counts({})

    assert counts is None
    assert are_tool_calls is False
    assert _compute_validation_turn_metrics([], sample_turns_are_tool_calls=False) == {}
