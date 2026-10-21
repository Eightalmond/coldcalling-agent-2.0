"""Step 5: turn-taking policies layered on Pipecat's user-turn machinery.

1. Backchannel-proof barge-in (`InterruptPhraseStartStrategy`).
   By default any detected voice starts a user turn, and a user turn
   interrupts the bot, so "mm-hmm", "yeah" or a cough cut Alex off. Instead,
   while Alex is speaking the caller must say INTERRUPT_MIN_WORDS words
   (Pipecat's MinWordsUserTurnStartStrategy, counting interim transcripts),
   except for short interrupt phrases like "wait" or "hold on", which still
   cut in immediately. When Alex is silent, one word starts a turn as usual.

2. Call pacing (`CallPacing`), driven by the user aggregator's events.
   - "Hold on" / "one second": the LLM acknowledges once (prompt rule), and
     the call switches to a long idle timeout so Alex just waits. The filler
     stays quiet too (see `CallPacing.on_hold`).
   - Silence: Pipecat's idle timer starts when Alex stops speaking. When it
     fires, Alex checks in ("Are you still there?"), at most
     MAX_IDLE_CHECKINS times, then says goodbye and ends the call.
"""

import random
import re
from collections.abc import Awaitable, Callable

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserIdleTimeoutUpdateFrame,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    UserTurnStoppedMessage,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

# While Alex is speaking, this many words are needed to interrupt.
INTERRUPT_MIN_WORDS = 3
# ...unless the caller starts with one of these (matched as whole words).
INTERRUPT_PHRASES: tuple[str, ...] = (
    "wait",
    "stop",
    "hold on",
    "hang on",
    "one second",
    "one sec",
    "excuse me",
    "sorry",
    "no",
    "actually",
)

# "Hold on" style requests. Only short utterances count, so questions like
# "does setup take one minute?" aren't mistaken for a request to wait.
_HOLD_RE = re.compile(
    r"\b(hold on|hang on|bear with me|"
    r"(one|just a|give me a|wait a) (sec|second|moment|minute))\b",
    re.IGNORECASE,
)
HOLD_MAX_WORDS = 6

IDLE_CHECKIN_SECS = 10.0  # silence before "Are you still there?"
HOLD_IDLE_SECS = 30.0  # silence allowed after "hold on" before a check-in
MAX_IDLE_CHECKINS = 2  # then say goodbye and end the call

CHECKIN_PHRASES: tuple[str, ...] = (
    "Are you still there?",
    "Just checking in, are you still with me?",
)
HOLD_CHECKIN_PHRASES: tuple[str, ...] = (
    "Take your time, I'm still here.",
    "No rush, I'm still on the line.",
)
GOODBYE_TEXT = (
    "It sounds like now isn't a good time. Thanks for your time, and have a great day. Goodbye!"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def starts_with_interrupt_phrase(text: str) -> bool:
    """True if the utterance begins with an interrupt phrase ("Wait, ..." / "Hold on")."""
    words = _words(text)
    return any(words[: len(p.split())] == p.split() for p in INTERRUPT_PHRASES)


def is_hold_request(text: str) -> bool:
    """True if the caller is asking Alex to wait ("hold on", "give me a second")."""
    return len(_words(text)) <= HOLD_MAX_WORDS and bool(_HOLD_RE.search(text))


class InterruptPhraseStartStrategy(MinWordsUserTurnStartStrategy):
    """MinWords start strategy that also lets short interrupt phrases through."""

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if starts_with_interrupt_phrase(frame.text):
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        # Everything else (word counting, bot speaking state) is MinWords' job.
        return await super().process_frame(frame)


QueueFrames = Callable[[list[Frame]], Awaitable[None]]


class CallPacing:
    """Hold handling and idle check-ins for one call.

    `queue_frames` is the PipelineWorker's queue_frames: frames enter at the
    start of the pipeline, so the idle-timeout updates reach the user
    aggregator and the spoken lines reach TTS.
    """

    def __init__(self, queue_frames: QueueFrames) -> None:
        self._queue_frames = queue_frames
        self._on_hold = False
        self._checkins = 0

    @property
    def on_hold(self) -> bool:
        return self._on_hold

    def attach(self, user_aggregator: LLMUserAggregator) -> None:
        user_aggregator.add_event_handler("on_user_turn_started", self._on_user_turn_started)
        user_aggregator.add_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)
        user_aggregator.add_event_handler("on_user_turn_idle", self._on_user_idle)

    async def _on_user_turn_started(self, aggregator: LLMUserAggregator, strategy: object) -> None:
        self._checkins = 0
        if self._on_hold:
            self._on_hold = False
            logger.info("Caller is back; leaving hold")
            await self._queue_frames([UserIdleTimeoutUpdateFrame(timeout=IDLE_CHECKIN_SECS)])

    async def _on_user_turn_stopped(
        self, aggregator: LLMUserAggregator, strategy: object, message: UserTurnStoppedMessage
    ) -> None:
        if message.content and is_hold_request(message.content):
            self._on_hold = True
            logger.info(f"Caller asked to hold; waiting up to {HOLD_IDLE_SECS:.0f}s")
            await self._queue_frames([UserIdleTimeoutUpdateFrame(timeout=HOLD_IDLE_SECS)])

    async def _on_user_idle(self, aggregator: LLMUserAggregator) -> None:
        if self._checkins >= MAX_IDLE_CHECKINS:
            logger.info("Caller silent after check-ins; saying goodbye and ending the call")
            # EndFrame queues behind the goodbye audio, so it plays in full first.
            await self._queue_frames([TTSSpeakFrame(GOODBYE_TEXT), EndFrame()])
            return
        self._checkins += 1
        phrases = HOLD_CHECKIN_PHRASES if self._on_hold else CHECKIN_PHRASES
        logger.info(f"Caller idle; check-in {self._checkins}/{MAX_IDLE_CHECKINS}")
        # Stored in context (the caller heard it), so the LLM knows it asked.
        await self._queue_frames([TTSSpeakFrame(random.choice(phrases))])
