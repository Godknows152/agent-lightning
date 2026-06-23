from agentlightning.verl.daemon import get_trajectory_prompt_delta_ids


def test_prompt_delta_uses_exact_context_prefix() -> None:
    base_prompt = [1, 2, 3]
    current_context = base_prompt + [10, 11]
    next_prompt = current_context + [20, 21, 22]

    delta, matched, truncated = get_trajectory_prompt_delta_ids(
        next_prompt,
        current_context,
        base_prompt,
        max_unmatched_prompt_delta_length=2,
    )

    assert delta == [20, 21, 22]
    assert matched is True
    assert truncated is False


def test_prompt_delta_removes_repeated_base_prompt_when_context_prefix_fails() -> None:
    base_prompt = list(range(1000))
    previous_response = [9000, 9001]
    current_context = base_prompt + previous_response
    new_state = list(range(2000, 2900))
    standalone_next_prompt = base_prompt + new_state

    delta, matched, truncated = get_trajectory_prompt_delta_ids(
        standalone_next_prompt,
        current_context,
        base_prompt,
        max_unmatched_prompt_delta_length=512,
    )

    assert matched is False
    assert truncated is True
    assert delta == new_state[-512:]
    assert len(delta) < len(standalone_next_prompt)


def test_prompt_delta_uses_suffix_prefix_overlap_before_bounded_fallback() -> None:
    base_prompt = [1, 2, 3]
    current_context = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    next_prompt = [52, 53, 54, 55, 56, 57, 58, 59, 70, 71]

    delta, matched, truncated = get_trajectory_prompt_delta_ids(
        next_prompt,
        current_context,
        base_prompt,
        max_unmatched_prompt_delta_length=1,
        min_overlap_tokens=4,
    )

    assert matched is True
    assert truncated is False
    assert delta == [70, 71]
