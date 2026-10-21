"""Offline tests for calle.metrics.TurnLatencyObserver, using simulated frames."""

import asyncio

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.tests.utils import SleepFrame, run_test

from calle.metrics import TurnLatencyObserver, TurnMetrics


def audio(context_id: str) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(b"\0" * 320, 16000, 1, context_id=context_id)


def transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "now")


def bot_reply(text: str, context_id: str = "answer") -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text),
        LLMFullResponseEndFrame(),
        audio(context_id),
    ]


def observe(frames: list[Frame]) -> list[TurnMetrics]:
    observer = TurnLatencyObserver()
    measured: list[TurnMetrics] = []

    @observer.event_handler("on_turn_metrics")
    async def collect(_observer: TurnLatencyObserver, metrics: TurnMetrics) -> None:
        measured.append(metrics)

    asyncio.run(run_test(IdentityFilter(), frames_to_send=frames, observers=[observer]))
    return measured


def test_normal_turn_is_anchored_on_vad_stop() -> None:
    [m] = observe(
        [
            transcript("How much is it?"),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
            SleepFrame(0.05),
            UserStoppedSpeakingFrame(),
            *bot_reply("15 dollars."),
        ]
    )
    assert m.anchor == "VAD stop"
    assert m.vad_wait_ms == 200
    assert m.turn_end_ms is not None and m.turn_end_ms >= 40
    assert m.total_ms is not None and m.total_ms >= m.turn_end_ms


def test_turn_without_vad_stop_falls_back_to_turn_end() -> None:
    # First half of a split turn consumes the VAD stop; the second half
    # (started by a transcript alone) has none of its own.
    turns = observe(
        [
            transcript("I am wondering if"),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
            UserStoppedSpeakingFrame(),
            *bot_reply("Sure."),
            transcript("it works for recruiters."),
            UserStoppedSpeakingFrame(),
            SleepFrame(0.05),
            *bot_reply("It does."),
        ]
    )
    second = turns[1]
    assert second.user_text == "it works for recruiters."
    assert second.anchor == "turn end"
    assert second.vad_wait_ms is None and second.stt_ms is None
    assert second.turn_end_ms == 0
    assert second.first_token_ms is not None and second.total_ms is not None


def test_empty_transcript_turn_is_not_logged() -> None:
    assert observe([VADUserStoppedSpeakingFrame(stop_secs=0.2), UserStoppedSpeakingFrame()]) == []


def test_filler_audio_is_not_counted_as_first_audio() -> None:
    [m] = observe(
        [
            transcript("Tell me about security."),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
            UserStoppedSpeakingFrame(),
            LLMFullResponseStartFrame(),
            TTSSpeakFrame("One moment.", append_to_context=False),
            audio("filler"),
            SleepFrame(0.1),
            LLMTextFrame("It is encrypted."),
            audio("filler"),  # trailing filler chunk after the first token
            SleepFrame(0.05),
            LLMFullResponseEndFrame(),
            audio("answer"),
        ]
    )
    assert m.filler_ms is not None
    assert m.token_to_audio_ms is not None and m.token_to_audio_ms >= 40
    assert m.bot_text == "It is encrypted."
