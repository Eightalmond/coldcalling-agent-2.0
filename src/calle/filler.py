"""Speak a short filler ("One moment.") when the LLM is slow to answer.

Sits between the LLM and TTS. After the user's turn ends, if the LLM hasn't
produced any text within `delay_secs`, push a filler TTSSpeakFrame. At most
one filler per user turn; cancelled if the LLM answers or the user speaks.

The filler is NOT stored in the conversation history: it's sent with
`append_to_context=False`, so the TTS marks its text "don't record" and the
assistant aggregator skips it. The LLM never sees it as something it said.
"""

import asyncio
import random

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

FILLER_PHRASES: tuple[str, ...] = (
    "One moment.",
    "Let me think about that.",
    "Good question, one second.",
    "Let me check that for you.",
)


class SlowResponseFiller(FrameProcessor):
    def __init__(self, delay_secs: float, phrases: tuple[str, ...] = FILLER_PHRASES) -> None:
        super().__init__()
        self._delay_secs = delay_secs
        self._phrases = phrases
        self._last_phrase = ""
        # True from the end of a user turn until the LLM answers or we fill.
        # The bot's opening greeting has no user turn, so it never gets a filler.
        self._awaiting_answer = False
        self._timer: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._awaiting_answer = True
        elif isinstance(frame, LLMFullResponseStartFrame) and self._awaiting_answer:
            # Also re-arms for an LLMErrorRecovery retry, which starts a new response.
            await self._start_timer()
        elif isinstance(frame, LLMTextFrame):
            self._awaiting_answer = False
            await self._cancel_timer()
        elif isinstance(frame, (UserStartedSpeakingFrame, InterruptionFrame)):
            self._awaiting_answer = False
            await self._cancel_timer()

        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._cancel_timer()

    async def _start_timer(self) -> None:
        await self._cancel_timer()
        self._timer = self.create_task(self._speak_after_delay())

    async def _cancel_timer(self) -> None:
        if self._timer:
            await self.cancel_task(self._timer)
            self._timer = None

    async def _speak_after_delay(self) -> None:
        await asyncio.sleep(self._delay_secs)
        self._timer = None
        if not self._awaiting_answer:
            return
        self._awaiting_answer = False  # one filler per user turn
        await self.push_frame(TTSSpeakFrame(self._pick_phrase(), append_to_context=False))

    def _pick_phrase(self) -> str:
        # Vary the wording, and never repeat the previous filler back-to-back.
        choices = [p for p in self._phrases if p != self._last_phrase] or list(self._phrases)
        self._last_phrase = random.choice(choices)
        return self._last_phrase
