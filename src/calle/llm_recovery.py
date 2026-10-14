"""Retry a failed LLM response once, then fall back to a spoken apology.

Why this exists: gpt-oss on Groq occasionally emits a tool call even though
no tools are defined, and Groq rejects it with HTTP 400 "Tool choice is none,
but model called a tool". Pipecat reports that as a non-fatal error and the
bot just goes silent until the user speaks again.

How it works (sits between the LLM and TTS):
- The LLM's `on_error` event tells us the current response failed.
- When that response's LLMFullResponseEndFrame passes through, we either
  retry by pushing the same context *upstream* to the LLM (the same thing
  Pipecat's assistant aggregator does for an LLMRunFrame), or, if we
  already retried this turn, speak a short fallback line.

Context history stays clean: a failed attempt produces no text, so the
assistant aggregator adds nothing; the retry reuses the unchanged context,
so the user's message isn't duplicated.
"""

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService

from calle.metrics import LLMErrorStats, log_llm_error

FALLBACK_TEXT = "Sorry, I lost my train of thought. Could you say that again?"


class LLMErrorRecovery(FrameProcessor):
    """Retries a failed LLM response once per user turn, then apologizes."""

    def __init__(self, llm: LLMService, context: LLMContext, max_retries: int = 1) -> None:
        super().__init__()
        self._llm = llm
        self._context = context
        self._max_retries = max_retries
        self._retries_this_turn = 0
        self._failed = False  # did the response currently streaming fail?
        self._got_text = False  # did it produce any text before failing?
        self._last_error = ""
        self.stats = LLMErrorStats()
        llm.add_event_handler("on_error", self._on_llm_error)

    async def _on_llm_error(self, llm: LLMService, error: ErrorFrame) -> None:
        self._failed = True
        self._last_error = error.error
        # Pipecat files any HTTP 400 under a "permanent" error category and
        # flags the service unusable. A bad tool call is a one-off bad
        # generation, not a broken request, so keep the LLM in service.
        await llm.set_usable(True)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._retries_this_turn = 0  # a new user turn gets a fresh retry
        elif isinstance(frame, LLMFullResponseStartFrame):
            # Don't reset _failed here: on_error runs in the LLM's own task and
            # can fire before we process this Start frame. Only End clears it.
            self._got_text = False
        elif isinstance(frame, LLMTextFrame):
            self._got_text = True

        await self.push_frame(frame, direction)

        # Act after passing the End frame on, so downstream sees the failed
        # response close before anything new starts.
        if isinstance(frame, LLMFullResponseEndFrame) and self._failed:
            self._failed = False
            await self._recover()

    async def _recover(self) -> None:
        self.stats.errors += 1
        if self._got_text:
            # Part of the reply was already spoken (and stored in context);
            # retrying would repeat it. Let the conversation continue.
            action = "gave up (partial reply already spoken)"
        elif self._retries_this_turn < self._max_retries:
            self._retries_this_turn += 1
            self.stats.retries += 1
            action = "retrying"
            await self.push_frame(LLMContextFrame(context=self._context), FrameDirection.UPSTREAM)
        else:
            self.stats.fallbacks += 1
            action = "speaking fallback"
            # Stored in context on purpose: the caller heard it, so the LLM
            # should know it asked them to repeat themselves.
            await self.push_frame(TTSSpeakFrame(FALLBACK_TEXT))
        log_llm_error(self._last_error, action, self.stats)
        logger.debug(f"{self}: LLM error handled: {action}")
