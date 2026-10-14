"""Per-turn latency measurement, reusable across every step of CALL-E.

`TurnLatencyObserver` is a Pipecat *observer*: it sees every frame that moves
between processors without being a stage in the pipeline, so it can't slow
down or alter the audio path. Attach it with `PipelineWorker(observers=[...])`.

What it measures, per user turn (all relative to the VAD "user stopped
speaking" moment):

    user speaks ... silence ──► VAD stop ──► final transcript ──► first TTS audio
                    └ vad_wait ┘           └──── stt_ms ─────┘
                                           └────────────── response_ms ────────┘

`vad_wait` is real latency the user feels but no timer here can see: Silero
must hear `stop_secs` of silence before it declares the turn over.
"""

from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class TurnMetrics:
    """Timings and text for one completed user -> bot exchange."""

    user_text: str
    bot_text: str
    vad_wait_ms: float
    # Signed: negative means Deepgram had the final transcript *before* the
    # VAD declared the stop, i.e. the VAD was the bottleneck, not STT.
    stt_ms: float | None
    response_ms: float


class TurnLatencyObserver(BaseObserver):
    """Logs one summary line per turn and emits an `on_turn_metrics` event.

    Subscribe from other code (e.g. the step-6 web UI latency readout) with:

        @observer.event_handler("on_turn_metrics")
        async def handle(observer, metrics: TurnMetrics): ...
    """

    def __init__(self) -> None:
        # A frame is re-pushed by every processor it passes through; we only
        # want the moment it was *created*, so observe first pushes only.
        super().__init__(observe_every_push=False)
        self._register_event_handler("on_turn_metrics")
        self._reset()

    def _reset(self) -> None:
        self._stopped_at: int | None = None  # pipeline-clock ns
        self._vad_wait_secs = 0.0
        self._transcript_at: int | None = None
        self._user_parts: list[str] = []
        self._bot_parts: list[str] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # The VAD broadcasts this both up- and downstream; count it once.
            # If the user pauses and resumes, the latest stop wins.
            if data.direction == FrameDirection.DOWNSTREAM:
                self._stopped_at = data.timestamp
                self._vad_wait_secs = frame.stop_secs

        elif isinstance(frame, TranscriptionFrame):
            self._user_parts.append(frame.text)
            self._transcript_at = data.timestamp

        elif isinstance(frame, TTSSpeakFrame):
            self._bot_parts.append(frame.text)

        elif isinstance(frame, TTSAudioRawFrame) and self._stopped_at is not None:
            # First audio chunk since the user stopped: the turn is answered.
            await self._finish_turn(first_audio_at=data.timestamp)

    async def _finish_turn(self, first_audio_at: int) -> None:
        assert self._stopped_at is not None
        stt_ms = None
        if self._transcript_at is not None:
            stt_ms = (self._transcript_at - self._stopped_at) / NS_PER_MS

        metrics = TurnMetrics(
            user_text=" ".join(self._user_parts),
            bot_text=" ".join(self._bot_parts),
            vad_wait_ms=self._vad_wait_secs * 1000,
            stt_ms=stt_ms,
            response_ms=(first_audio_at - self._stopped_at) / NS_PER_MS,
        )
        self._reset()

        log_turn(metrics)
        await self._call_event_handler("on_turn_metrics", metrics)


def log_turn(metrics: TurnMetrics) -> None:
    """Print a compact, readable block for one turn."""
    stt = "n/a" if metrics.stt_ms is None else f"{metrics.stt_ms:+.0f} ms"
    logger.info(
        "\n── turn ─────────────────────────────\n"
        f"  user : {metrics.user_text!r}\n"
        f"  bot  : {metrics.bot_text!r}\n"
        f"  vad wait {metrics.vad_wait_ms:.0f} ms | stt {stt} | "
        f"response {metrics.response_ms:.0f} ms (VAD stop -> first audio)"
    )
