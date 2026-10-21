"""Offline tests for calle.guardrails and calle.call_timer (fake clock, no network)."""

import asyncio

import pytest

from pipecat.frames.frames import EndFrame, Frame, TTSSpeakFrame

from calle.call_timer import WRAP_UP_TEXT, CallTimer
from calle.guardrails import (
    CALL_CONNECT_TIMEOUT_SECS,
    CALLS_PER_DAY,
    CALLS_PER_VISITOR_PER_HOUR,
    HOUR,
    CallRegistry,
    LimitExceeded,
    SlidingWindowLimiter,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_sliding_window_allows_up_to_limit_then_waits() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window_secs=60, clock=clock)
    for _ in range(2):
        assert limiter.retry_after("a") == 0
        limiter.record("a")
    assert limiter.retry_after("a") == pytest.approx(60)
    assert limiter.retry_after("b") == 0  # other visitors unaffected
    clock.now += 61
    assert limiter.retry_after("a") == 0


def test_per_visitor_hourly_limit() -> None:
    clock = FakeClock()
    registry = CallRegistry(clock=clock, max_concurrent=100)
    for _ in range(CALLS_PER_VISITOR_PER_HOUR):
        registry.finish(registry.create("brief", "1.2.3.4").call_id)
    with pytest.raises(LimitExceeded, match="this hour") as error:
        registry.create("brief", "1.2.3.4")
    assert error.value.retry_after > 0
    registry.create("brief", "5.6.7.8")  # someone else can still call
    clock.now += HOUR + 1
    registry.create("brief", "1.2.3.4")


def test_daily_limit_across_all_visitors() -> None:
    registry = CallRegistry(clock=FakeClock(), max_concurrent=1000)
    for i in range(CALLS_PER_DAY):
        registry.finish(registry.create("brief", f"visitor-{i}").call_id)
    with pytest.raises(LimitExceeded, match="today"):
        registry.create("brief", "new-visitor")


def test_concurrent_call_limit() -> None:
    registry = CallRegistry(clock=FakeClock(), max_concurrent=2)
    first = registry.create("brief", "a")
    registry.create("brief", "b")
    with pytest.raises(LimitExceeded, match="busy"):
        registry.create("brief", "c")
    registry.finish(first.call_id)
    registry.create("brief", "c")


def test_call_id_is_one_time_and_expires() -> None:
    clock = FakeClock()
    registry = CallRegistry(clock=clock)
    call = registry.create("brief", "a")
    assert registry.claim(call.call_id) is call
    assert registry.claim(call.call_id) is None  # already used
    assert registry.claim("made-up-id") is None

    stale = registry.create("brief", "b")
    clock.now += CALL_CONNECT_TIMEOUT_SECS + 1
    assert registry.claim(stale.call_id) is None  # never connected in time
    assert registry.open_calls == 1  # only the connected call remains


def test_call_timer_wraps_up_then_cancels() -> None:
    queued: list[list[Frame]] = []
    cancelled: list[bool] = []

    async def queue_frames(frames: list[Frame]) -> None:
        queued.append(frames)

    async def cancel() -> None:
        cancelled.append(True)

    async def scenario() -> None:
        timer = CallTimer(0.05, queue_frames, cancel, grace_secs=0.05)
        timer.start()
        await asyncio.sleep(0.2)
        await timer.stop()

    asyncio.run(scenario())
    [frames] = queued
    assert isinstance(frames[0], TTSSpeakFrame) and frames[0].text == WRAP_UP_TEXT
    assert isinstance(frames[1], EndFrame)
    assert cancelled == [True]


def test_call_timer_stop_before_limit_does_nothing() -> None:
    queued: list[list[Frame]] = []

    async def queue_frames(frames: list[Frame]) -> None:
        queued.append(frames)

    async def cancel() -> None:
        raise AssertionError("should not cancel")

    async def scenario() -> None:
        timer = CallTimer(0.5, queue_frames, cancel)
        timer.start()
        await asyncio.sleep(0.05)
        await timer.stop()

    asyncio.run(scenario())
    assert queued == []
