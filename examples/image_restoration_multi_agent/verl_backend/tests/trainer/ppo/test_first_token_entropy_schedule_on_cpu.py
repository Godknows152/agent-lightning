import pytest

from verl.trainer.ppo.first_token_entropy_schedule import get_first_token_entropy_coeff


def test_constant_schedule_preserves_legacy_coefficient() -> None:
    values = [
        get_first_token_entropy_coeff(
            step=step,
            total_steps=100,
            start=0.003,
            end=0.0003,
            schedule="constant",
        )
        for step in (1, 20, 80, 100)
    ]

    assert values == [pytest.approx(0.003)] * 4


def test_wsd_cosine_schedule_has_ramp_stable_decay_and_floor() -> None:
    kwargs = {
        "total_steps": 101,
        "start": 1.0,
        "end": 0.1,
        "schedule": "wsd_cosine",
        "ramp_ratio": 0.10,
        "stable_end_ratio": 0.30,
        "decay_end_ratio": 0.80,
    }

    assert get_first_token_entropy_coeff(step=1, **kwargs) == pytest.approx(0.0)
    assert get_first_token_entropy_coeff(step=11, **kwargs) == pytest.approx(1.0)
    assert get_first_token_entropy_coeff(step=30, **kwargs) == pytest.approx(1.0)
    assert 0.1 < get_first_token_entropy_coeff(step=55, **kwargs) < 1.0
    assert get_first_token_entropy_coeff(step=81, **kwargs) == pytest.approx(0.1)
    assert get_first_token_entropy_coeff(step=101, **kwargs) == pytest.approx(0.1)


def test_wsd_cosine_schedule_clamps_resume_steps() -> None:
    kwargs = {
        "total_steps": 10,
        "start": 0.003,
        "end": 0.0003,
        "schedule": "wsd_cosine",
    }

    assert get_first_token_entropy_coeff(step=0, **kwargs) == pytest.approx(0.0)
    assert get_first_token_entropy_coeff(step=100, **kwargs) == pytest.approx(0.0003)


def test_wsd_cosine_schedule_can_decay_immediately_after_warmup() -> None:
    kwargs = {
        "total_steps": 101,
        "start": 1.0,
        "end": 0.1,
        "schedule": "wsd_cosine",
        "ramp_ratio": 0.05,
        "stable_end_ratio": 0.05,
        "decay_end_ratio": 1.0,
    }

    assert get_first_token_entropy_coeff(step=6, **kwargs) == pytest.approx(1.0)
    assert get_first_token_entropy_coeff(step=7, **kwargs) < 1.0
    assert get_first_token_entropy_coeff(step=101, **kwargs) == pytest.approx(0.1)


def test_schedule_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="ratios"):
        get_first_token_entropy_coeff(
            step=1,
            total_steps=10,
            start=0.003,
            schedule="wsd_cosine",
            ramp_ratio=0.4,
            stable_end_ratio=0.2,
        )

    with pytest.raises(ValueError, match="must not exceed"):
        get_first_token_entropy_coeff(
            step=1,
            total_steps=10,
            start=0.003,
            end=0.004,
            schedule="wsd_cosine",
        )
