"""Per-turn latency measurement, reusable across every step of CALL-E.

`TurnLatencyObserver` is a Pipecat *observer*: it sees every frame that moves
between processors without being a stage in the pipeline, so it can't slow
down or alter the audio path. Attach it with `PipelineWorker(observers=[...])`.

Timeline of one turn (all times relative to the VAD "user stopped" moment):

    speech ... silence ─► VAD stop ─► final transcript ─► turn end ─► first LLM token ─► first TTS audio
              └vad_wait┘           └── stt ──────────┘
                                   └── turn_end ─────────────────┘
                                   └── first_token ───────────────────────────────┘
                                                                                  └ token_to_audio ┘
                                   └── total ──────────────────────────────────────────────────────┘

- vad_wait: silence Silero must hear before it declares a stop. Real latency
  the user feels, but it happens before our clock starts.
- stt: signed; negative means Deepgram had the final transcript before the
  VAD stop (the VAD was the bottleneck).
- turn_end: when the user aggregator decided the turn was over (Smart Turn
  model + waiting for the transcript). The LLM request starts here.
- token_to_audio: TTS buffers LLM tokens into a full sentence before it
  synthesizes, so this includes sentence aggregation, not just TTS speed.
- filler: when the LLM is slow, `SlowResponseFiller` speaks "One moment."
  first. That audio is not the answer, so it doesn't count as first audio;
  the filler's own timing is logged separately.

LLM errors (see `llm_recovery.py`) are logged by `log_llm_error` with running
counters, so we can see how often retries and fallbacks happen.
"""

from dataclasses import dataclass, field

from loguru import logger

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

NS_PER_MS = 1_000_000
MAX_ERROR_CHARS = 200


@dataclass
class LLMErrorStats:
    """Running counts for this call, so error frequency is measurable."""

    errors: int = 0
    retries: int = 0
    fallbacks: int = 0


def log_llm_error(error: str, action: str, stats: LLMErrorStats) -> None:
    """One WARNING line per LLM failure. Logs the error text, never the prompt."""
    short = error if len(error) <= MAX_ERROR_CHARS else error[:MAX_ERROR_CHARS] + "..."
    logger.warning(
        f"LLM error ({action}) | errors={stats.errors} retries={stats.retries} "
        f"fallbacks={stats.fallbacks} | {short}"
    )


@dataclass(frozen=True)
class TurnMetrics:
    """Timings (ms) and text for one completed user -> bot exchange.

    Timings are None when the moment never happened, e.g. the user
    interrupted before any audio was produced.
    """

    user_text: str
    bot_text: str
    interrupted: bool
    vad_wait_ms: float
    stt_ms: float | None
    turn_end_ms: float | None
    first_token_ms: float | None
    token_to_audio_ms: float | None
    total_ms: float | None
    filler_ms: float | None = None  # VAD stop -> filler requested, if one was


@dataclass
class _OpenTurn:
    """Mutable bookkeeping for a turn waiting on the bot's response."""

    user_text: str
    vad_stop_at: int | None  # all timestamps: pipeline-clock ns
    vad_wait_secs: float
    stt_at: int | None
    turn_end_at: int
    first_token_at: int | None = None
    first_audio_at: int | None = None
    filler_at: int | None = None
    # When the real answer began: first LLM token, or the fallback line.
    answer_started_at: int | None = None
    # TTS context ids of audio heard before the answer began (the filler), so
    # its trailing chunks aren't mistaken for the answer's first audio.
    pre_answer_contexts: set[str] = field(default_factory=set)
    llm_done: bool = False
    bot_parts: list[str] = field(default_factory=list)


def _ms_between(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) / NS_PER_MS


class TurnLatencyObserver(BaseObserver):
    """Logs one summary block per turn and emits an `on_turn_metrics` event.

    Subscribe from other code (e.g. the step-6 web UI latency readout) with:

        @observer.event_handler("on_turn_metrics")
        async def handle(observer, metrics: TurnMetrics): ...
    """

    def __init__(self) -> None:
        # A frame is re-pushed by every processor it passes through; we only
        # want the moment it was *created*, so observe first pushes only.
        super().__init__(observe_every_push=False)
        self._register_event_handler("on_turn_metrics")
        # Collected while the user is talking:
        self._vad_stop_at: int | None = None
        self._vad_wait_secs = 0.0
        self._stt_at: int | None = None
        self._user_parts: list[str] = []
        # Set once the user's turn has ended and we're timing the response:
        self._turn: _OpenTurn | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        frame, now = data.frame, data.timestamp

        # SystemFrames like these are broadcast both up- and downstream;
        # count each event once by only looking at the downstream copy.
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # If the user pauses and resumes, the latest stop wins.
            self._vad_stop_at = now
            self._vad_wait_secs = frame.stop_secs

        elif isinstance(frame, TranscriptionFrame):
            self._user_parts.append(frame.text.strip())
            self._stt_at = now

        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._on_user_turn_end(now)

        elif self._turn is None:
            return  # e.g. the bot's greeting: not a response to a user turn

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._turn.llm_done = False  # a retry starts a fresh response

        elif isinstance(frame, LLMTextFrame):
            if self._turn.first_token_at is None:
                self._turn.first_token_at = now
            if self._turn.answer_started_at is None:
                self._turn.answer_started_at = now
            self._turn.bot_parts.append(frame.text)

        elif isinstance(frame, TTSSpeakFrame):
            if not frame.append_to_context:  # the filler
                if self._turn.filler_at is None:
                    self._turn.filler_at = now
            else:  # the fallback line after a failed retry
                if self._turn.answer_started_at is None:
                    self._turn.answer_started_at = now
                self._turn.bot_parts.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._turn.llm_done = True
            await self._maybe_finish()

        elif isinstance(frame, TTSAudioRawFrame):
            await self._on_audio(frame, now)

        elif isinstance(frame, InterruptionFrame):
            await self._finish(interrupted=True)

    async def _on_user_turn_end(self, now: int) -> None:
        user_text = " ".join(part for part in self._user_parts if part)
        if not user_text:
            # VAD heard something (a cough, a click, background noise) but
            # STT found no words. Not a real turn.
            logger.debug("VAD blip: user turn ended with an empty transcript; not logged")
        else:
            if self._turn is not None:  # previous response never completed
                await self._finish(interrupted=True)
            self._turn = _OpenTurn(
                user_text=user_text,
                vad_stop_at=self._vad_stop_at,
                vad_wait_secs=self._vad_wait_secs,
                stt_at=self._stt_at,
                turn_end_at=now,
            )
        self._vad_stop_at = self._stt_at = None
        self._user_parts = []

    async def _on_audio(self, frame: TTSAudioRawFrame, now: int) -> None:
        assert self._turn is not None
        if self._turn.first_audio_at is not None:
            return
        if self._turn.answer_started_at is None:
            # Audio before any answer exists can only be the filler.
            if frame.context_id:
                self._turn.pre_answer_contexts.add(frame.context_id)
            return
        if frame.context_id in self._turn.pre_answer_contexts:
            return  # trailing filler audio
        self._turn.first_audio_at = now
        await self._maybe_finish()

    async def _maybe_finish(self) -> None:
        # Log once the bot's full text is known AND its first audio went out;
        # these can arrive in either order.
        if self._turn and self._turn.llm_done and self._turn.first_audio_at is not None:
            await self._finish(interrupted=False)

    async def _finish(self, *, interrupted: bool) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        metrics = TurnMetrics(
            user_text=turn.user_text,
            bot_text="".join(turn.bot_parts).strip(),
            interrupted=interrupted,
            vad_wait_ms=turn.vad_wait_secs * 1000,
            stt_ms=_ms_between(turn.vad_stop_at, turn.stt_at),
            turn_end_ms=_ms_between(turn.vad_stop_at, turn.turn_end_at),
            first_token_ms=_ms_between(turn.vad_stop_at, turn.first_token_at),
            token_to_audio_ms=_ms_between(turn.first_token_at, turn.first_audio_at),
            total_ms=_ms_between(turn.vad_stop_at, turn.first_audio_at),
            filler_ms=_ms_between(turn.vad_stop_at, turn.filler_at),
        )
        log_turn(metrics)
        await self._call_event_handler("on_turn_metrics", metrics)


def _fmt(ms: float | None, signed: bool = False) -> str:
    if ms is None:
        return "n/a"
    # Round first so -0.3 prints as "+0", not "-0".
    return f"{round(ms):+d} ms" if signed else f"{round(ms)} ms"


def log_turn(metrics: TurnMetrics) -> None:
    """Print a compact, readable block for one turn."""
    status = " (interrupted)" if metrics.interrupted else ""
    filler = f" | filler at {_fmt(metrics.filler_ms)}" if metrics.filler_ms is not None else ""
    logger.info(
        f"\n── turn{status} ─────────────────────────────\n"
        f"  user : {metrics.user_text!r}\n"
        f"  bot  : {metrics.bot_text!r}\n"
        f"  vad wait {_fmt(metrics.vad_wait_ms)} | stt {_fmt(metrics.stt_ms, signed=True)}"
        f" | turn end {_fmt(metrics.turn_end_ms, signed=True)}\n"
        f"  VAD stop -> first token {_fmt(metrics.first_token_ms)}"
        f" | first token -> first audio {_fmt(metrics.token_to_audio_ms)}"
        f" | total {_fmt(metrics.total_ms)}{filler}"
    )
