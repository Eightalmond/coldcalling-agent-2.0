"""Offline tests for calle.turn_taking (and the filler's hold hook)."""

import asyncio
from collections.abc import Callable

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TTSSpeakFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from calle.filler import SlowResponseFiller
from calle.turn_taking import (
    CHECKIN_PHRASES,
    GOODBYE_TEXT,
    HOLD_CHECKIN_PHRASES,
    HOLD_IDLE_SECS,
    IDLE_CHECKIN_SECS,
    INTERRUPT_MIN_WORDS,
    MAX_IDLE_CHECKINS,
    CallPacing,
    BargeInStartStrategy,
    is_hold_request,
    starts_with_interrupt_phrase,
)

# ---------------------------------------------------------------- phrase matching


@pytest.mark.parametrize("text", ["Wait.", "wait, what about pricing", "Hold on", "Sorry?", "No, I"])
def test_interrupt_phrases(text: str) -> None:
    assert starts_with_interrupt_phrase(text)


@pytest.mark.parametrize("text", ["mm-hmm", "yeah", "okay", "I'll wait and see", "nobody"])
def test_not_interrupt_phrases(text: str) -> None:
    assert not starts_with_interrupt_phrase(text)


@pytest.mark.parametrize(
    "text", ["Hold on.", "Hang on a sec", "Give me a second.", "one moment please", "Bear with me."]
)
def test_hold_requests(text: str) -> None:
    assert is_hold_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "Can I get a second opinion?",
        "Does setup really take just one minute or longer for a big team?",
        "How much does it cost?",
    ],
)
def test_not_hold_requests(text: str) -> None:
    assert not is_hold_request(text)


# ---------------------------------------------------------------- barge-in with the real aggregator


def run_user_aggregator(frames: list[Frame], context: LLMContext | None = None) -> list[Frame]:
    """Push frames through Pipecat's LLMUserAggregator using our start strategy."""
    pair = LLMContextAggregatorPair(
        context or LLMContext(),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[BargeInStartStrategy(min_words=INTERRUPT_MIN_WORDS)],
                # A simple stop strategy, so the test doesn't load the Smart Turn model.
                stop=[SpeechTimeoutUserTurnStopStrategy()],
            )
        ),
    )
    down, _ = asyncio.run(run_test(pair.user(), frames_to_send=frames))
    return list(down)


def interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text, "user", "now")


def interrupted(frames: list[Frame]) -> bool:
    return any(isinstance(f, UserStartedSpeakingFrame) for f in frames) and any(
        isinstance(f, InterruptionFrame) for f in frames
    )


def test_backchannel_does_not_interrupt_bot() -> None:
    frames = run_user_aggregator([BotStartedSpeakingFrame(), interim("mm-hmm")])
    assert not interrupted(frames)


def test_two_words_do_not_interrupt_bot() -> None:
    frames = run_user_aggregator([BotStartedSpeakingFrame(), interim("yeah okay")])
    assert not interrupted(frames)


def test_three_words_interrupt_bot() -> None:
    frames = run_user_aggregator([BotStartedSpeakingFrame(), interim("what about pricing")])
    assert interrupted(frames)


@pytest.mark.parametrize("phrase", ["Wait", "Hold on", "Sorry"])
def test_interrupt_phrase_cuts_in_immediately(phrase: str) -> None:
    frames = run_user_aggregator([BotStartedSpeakingFrame(), interim(phrase)])
    assert interrupted(frames)


def test_one_word_starts_a_turn_when_bot_is_silent() -> None:
    frames = run_user_aggregator(
        [BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), interim("Hello")]
    )
    assert any(isinstance(f, UserStartedSpeakingFrame) for f in frames)


def final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "now")


def user_messages(context: LLMContext) -> list[str]:
    return [m["content"] for m in context.messages if m.get("role") == "user"]


def test_words_count_across_segments_and_none_are_lost() -> None:
    # Live-test regression: Deepgram finalized "Okay." and "I'll give" as
    # separate segments; counted one by one, both were under 3 words and got
    # discarded, so the context only received "it a shot."
    context = LLMContext()
    frames = run_user_aggregator(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            # Spaced like real Deepgram segments. (Pushed all at once, the
            # interruption at turn start would flush the still-queued ones.)
            final("Okay."),
            SleepFrame(0.2),
            final("I'll give"),
            SleepFrame(0.2),
            final("it a shot."),
            SleepFrame(0.1),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(1.0),  # let the speech-timeout stop strategy end the turn
        ],
        context,
    )
    assert interrupted(frames)
    assert user_messages(context) == ["Okay. I'll give it a shot."]


def test_backchannel_is_dropped_and_does_not_leak_into_next_turn() -> None:
    context = LLMContext()
    run_user_aggregator(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            final("Mm-hmm."),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(1.2),  # a real pause: the next speech is a new utterance
            BotStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            final("What about pricing?"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(1.0),
        ],
        context,
    )
    assert user_messages(context) == ["What about pricing?"]


# ---------------------------------------------------------------- CallPacing


class FakeAggregator:
    """Stands in for LLMUserAggregator: records handlers so tests can fire events."""

    def __init__(self) -> None:
        self.handlers: dict[str, Callable] = {}

    def add_event_handler(self, name: str, handler: Callable) -> None:
        self.handlers[name] = handler

    def fire(self, name: str, *args: object) -> None:
        asyncio.run(self.handlers[name](self, *args))


def make_pacing() -> tuple[CallPacing, FakeAggregator, list[list[Frame]]]:
    queued: list[list[Frame]] = []

    async def queue_frames(frames: list[Frame]) -> None:
        queued.append(frames)

    pacing = CallPacing(queue_frames)
    aggregator = FakeAggregator()
    pacing.attach(aggregator)  # type: ignore[arg-type]
    return pacing, aggregator, queued


def user_said(text: str) -> UserTurnStoppedMessage:
    return UserTurnStoppedMessage(content=text, timestamp="now")


def test_idle_checkins_then_goodbye_and_end() -> None:
    _, aggregator, queued = make_pacing()
    for _ in range(MAX_IDLE_CHECKINS):
        aggregator.fire("on_user_turn_idle")
    aggregator.fire("on_user_turn_idle")

    checkins = queued[:MAX_IDLE_CHECKINS]
    assert all(isinstance(batch[0], TTSSpeakFrame) for batch in checkins)
    assert all(batch[0].text in CHECKIN_PHRASES for batch in checkins)
    goodbye = queued[-1]
    assert goodbye[0].text == GOODBYE_TEXT and isinstance(goodbye[1], EndFrame)


def test_caller_speaking_resets_checkins() -> None:
    _, aggregator, queued = make_pacing()
    aggregator.fire("on_user_turn_idle")
    aggregator.fire("on_user_turn_started", None)
    aggregator.fire("on_user_turn_idle")
    aggregator.fire("on_user_turn_idle")
    assert not any(isinstance(f, EndFrame) for batch in queued for f in batch)


def test_hold_extends_idle_timeout_and_uses_hold_checkins() -> None:
    pacing, aggregator, queued = make_pacing()
    aggregator.fire("on_user_turn_stopped", None, user_said("Hold on a second."))
    assert pacing.on_hold
    assert isinstance(queued[-1][0], UserIdleTimeoutUpdateFrame)
    assert queued[-1][0].timeout == HOLD_IDLE_SECS

    aggregator.fire("on_user_turn_idle")
    assert queued[-1][0].text in HOLD_CHECKIN_PHRASES

    aggregator.fire("on_user_turn_started", None)  # caller is back
    assert not pacing.on_hold
    assert queued[-1][0].timeout == IDLE_CHECKIN_SECS


def test_normal_question_is_not_a_hold() -> None:
    pacing, aggregator, queued = make_pacing()
    aggregator.fire("on_user_turn_stopped", None, user_said("How much does it cost?"))
    assert not pacing.on_hold and queued == []


# ---------------------------------------------------------------- filler hold hook


def filler_spoke(skip: bool) -> bool:
    filler = SlowResponseFiller(0.05, skip_if=lambda: skip)
    frames = [UserStoppedSpeakingFrame(), LLMFullResponseStartFrame(), SleepFrame(0.2)]
    down, _ = asyncio.run(run_test(filler, frames_to_send=frames))
    return any(isinstance(f, TTSSpeakFrame) for f in down)


def test_filler_speaks_when_not_on_hold() -> None:
    assert filler_spoke(skip=False)


def test_filler_is_silent_on_hold() -> None:
    assert not filler_spoke(skip=True)
