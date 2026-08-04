import asyncio

from verl.experimental.agent_loop.agent_loop import _GenerationToolPhaseCoordinator


def test_generation_and_tool_work_do_not_overlap_on_cpu():
    events: list[str] = []

    async def enter_tool_phase():
        events.append("sleep_sglang")

    async def enter_generation_phase():
        events.append("wake_sglang")

    async def run_test():
        coordinator = _GenerationToolPhaseCoordinator(
            2,
            on_tool_phase_start=enter_tool_phase,
            on_generation_phase_start=enter_generation_phase,
        )

        async def trajectory(name: str, generation_delay: float, tool_delay: float):
            events.append(f"generation_start:{name}")
            await asyncio.sleep(generation_delay)
            events.append(f"generation_end:{name}")
            await coordinator.after_generation()
            events.append(f"tool_start:{name}")
            await asyncio.sleep(tool_delay)
            events.append(f"tool_end:{name}")
            await coordinator.after_tool()
            events.append(f"next_generation_start:{name}")
            await coordinator.depart()

        await asyncio.gather(
            trajectory("fast", 0.01, 0.03),
            trajectory("slow", 0.03, 0.01),
        )

    asyncio.run(run_test())

    first_tool_start = min(index for index, event in enumerate(events) if event.startswith("tool_start:"))
    last_generation_end = max(index for index, event in enumerate(events) if event.startswith("generation_end:"))
    first_next_generation = min(
        index for index, event in enumerate(events) if event.startswith("next_generation_start:")
    )
    last_tool_end = max(index for index, event in enumerate(events) if event.startswith("tool_end:"))
    assert last_generation_end < events.index("sleep_sglang") < first_tool_start
    assert last_tool_end < events.index("wake_sglang") < first_next_generation


def test_trajectory_terminating_after_generation_does_not_deadlock_tool_phase_on_cpu():
    async def run_test():
        coordinator = _GenerationToolPhaseCoordinator(2)

        async def terminating_trajectory():
            await coordinator.after_generation()
            await coordinator.depart()

        async def tool_trajectory():
            await coordinator.after_generation()
            await asyncio.sleep(0.01)
            await coordinator.after_tool()
            await coordinator.depart()

        await asyncio.wait_for(
            asyncio.gather(terminating_trajectory(), tool_trajectory()),
            timeout=1.0,
        )

    asyncio.run(run_test())
