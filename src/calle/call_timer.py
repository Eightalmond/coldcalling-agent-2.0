"""End a call politely when it reaches its time limit.

At `max_secs`, Alex says a short wrap-up and an EndFrame is queued behind it
(the output transport plays the wrap-up fully before ending). If the call is
somehow still running `grace_secs` later, it is cancelled outright.
"""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from pipecat.frames.frames import EndFrame, Frame, TTSSpeakFrame

WRAP_UP_TEXT = (
    "We've reached the time limit for this demo call. "
    "Thanks so much for chatting with me. Goodbye!"
)
GRACE_SECS = 30.0

QueueFrames = Callable[[list[Frame]], Awaitable[None]]
Cancel = Callable[[], Awaitable[None]]


class CallTimer:
    def __init__(
        self,
        max_secs: float,
        queue_frames: QueueFrames,
        cancel: Cancel,
        grace_secs: float = GRACE_SECS,
    ) -> None:
        self._max_secs = max_secs
        self._queue_frames = queue_frames
        self._cancel = cancel
        self._grace_secs = grace_secs
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        await asyncio.sleep(self._max_secs)
        logger.info(f"Call reached its {self._max_secs:.0f}s limit; wrapping up")
        await self._queue_frames([TTSSpeakFrame(WRAP_UP_TEXT), EndFrame()])
        await asyncio.sleep(self._grace_secs)
        logger.warning("Call still running after wrap-up; cancelling")
        await self._cancel()
