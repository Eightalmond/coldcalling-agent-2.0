"""CALL-E step 3: an LLM sales persona.

    mic -> VAD -> Deepgram STT -> user context -> Groq LLM -> Deepgram TTS
        -> speaker -> assistant context

Run it (opens a browser client at http://localhost:7860/client):

    uv run python -m calle.bot

Why `python -m calle.bot` and not a script entry point: Pipecat's dev
runner finds your `bot()` function by looking at the `__main__` module.
`-m` makes this file `__main__`; a console-script wrapper would not.

Choose the product brief with `--brief briefs/<name>.md` or the CALLE_BRIEF
env var (flag wins). Without either, the built-in Quillo DEMO_BRIEF is used.
"""

import argparse
import os
from dataclasses import asdict

import certifi

# python.org builds of Python on macOS ship without trusted root certs, so
# Deepgram's websocket fails with CERTIFICATE_VERIFY_FAILED. Point OpenSSL at
# certifi's CA bundle. This must run before the pipecat imports below:
# aiohttp (used by pipecat) builds its SSL context once, at import time.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from calle.call_timer import CallTimer
from calle.config import ConfigError, load_settings
from calle.filler import SlowResponseFiller
from calle.knowledge import KnowledgeError, read_brief_file
from calle.llm_recovery import LLMErrorRecovery
from calle.log_redaction import install_log_redaction
from calle.metrics import TurnLatencyObserver, TurnMetrics
from calle.prompts import DEMO_BRIEF, GREETING_INSTRUCTION, build_system_prompt
from calle.turn_taking import (
    IDLE_CHECKIN_SECS,
    INTERRUPT_MIN_WORDS,
    CallPacing,
    InterruptPhraseStartStrategy,
)

# Aura-2 "Thalia": clear, upbeat American English. A good fit for a
# friendly sales rep. Other voices: https://developers.deepgram.com/docs/tts-models
TTS_VOICE = "aura-2-thalia-en"
STT_MODEL = "nova-3-general"

# gpt-oss-20b is a reasoning model: it "thinks" before its first spoken word.
# "low" is the lowest effort Groq accepts for gpt-oss (there is no "none").
# The reasoning text arrives in a separate field that Pipecat doesn't speak.
LLM_MODEL = "openai/gpt-oss-20b"
LLM_REASONING_EFFORT = "low"

# Smart Turn's silence fallback (SmartTurnParams.stop_secs, Pipecat default 3).
# When the model keeps judging the user "not finished", the turn still ends
# after this much silence ("End of Turn complete due to stop_secs" in the log).
# Lower = less dead air after a trailing "um..."; too low cuts off callers
# who pause to think mid-sentence.
SMART_TURN_STOP_SECS = 1.5

# If the LLM hasn't produced any text this long after the user's turn ends,
# say a short filler ("One moment.") so the caller isn't left in silence.
FILLER_DELAY_SECS = 1.5

BRIEF_ENV_VAR = "CALLE_BRIEF"


def resolve_brief(brief_path: str | None) -> tuple[str, str]:
    """Return (brief text, label for logs). Flag > env var > built-in demo.

    Logs only the brief's name and size, never its contents.
    Raises KnowledgeError if a given path is missing or empty.
    """
    path = brief_path or os.getenv(BRIEF_ENV_VAR)
    if not path:
        return DEMO_BRIEF, "built-in DEMO_BRIEF (Quillo)"
    brief, problems = read_brief_file(path)
    if problems:
        # Still usable (maybe hand-edited), but the prompt rules expect the
        # standard headings, e.g. "Not covered".
        logger.warning(f"Brief {os.path.basename(path)} has format problems: {'; '.join(problems)}")
    return brief, os.path.basename(path)


def add_brief_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--brief",
        metavar="PATH",
        help=f"product brief markdown file (default: ${BRIEF_ENV_VAR}, else the demo brief)",
    )


async def run_bot(
    transport: BaseTransport,
    brief: str,
    *,
    handle_sigint: bool = False,
    max_call_secs: float | None = None,
) -> None:
    """Run one call on `transport` until the client leaves (or the call cap hits).

    Shared by the Pipecat dev runner (`bot()` below) and our own server
    (`calle.server`), which also passes `max_call_secs`.
    """
    settings = load_settings()
    system_prompt = build_system_prompt(brief)
    install_log_redaction(secrets=settings.secrets(), system_prompt=system_prompt)

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(model=STT_MODEL),
    )
    tts = DeepgramTTSService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramTTSService.Settings(voice=TTS_VOICE),
    )
    llm = GroqLLMService(
        api_key=settings.groq_api_key,
        # The system prompt goes in system_instruction rather than as the
        # first context message, so it isn't re-logged with the context on
        # every LLM call.
        settings=GroqLLMService.Settings(
            model=LLM_MODEL,
            reasoning_effort=LLM_REASONING_EFFORT,
            system_instruction=system_prompt,
        ),
    )

    # The conversation history. The aggregators write to it: the user one
    # when a user turn ends, the assistant one with what the bot actually said.
    context = LLMContext()
    # No vad_analyzer here: our VADProcessor upstream already emits VAD frames,
    # and the aggregator's turn logic consumes those. Passing one would run a
    # second Silero model.
    # Turn start: while Alex talks, the caller needs INTERRUPT_MIN_WORDS words
    # (or an interrupt phrase like "wait") to cut in, so "mm-hmm" doesn't.
    # Turn stop: the bundled Smart Turn v3 model with a shorter silence fallback.
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=SMART_TURN_STOP_SECS)
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[InterruptPhraseStartStrategy(min_words=INTERRUPT_MIN_WORDS)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)],
            ),
            # Pipecat's idle timer: starts when Alex stops talking; CallPacing
            # handles the event (check-ins, then goodbye).
            user_idle_timeout=IDLE_CHECKIN_SECS,
        ),
    )
    # "Hold on" handling and idle check-ins. queue_frames is looked up when
    # called, so it can refer to the worker created below.
    pacing = CallPacing(queue_frames=lambda frames: worker.queue_frames(frames))
    pacing.attach(aggregators.user())

    pipeline = Pipeline(
        [
            transport.input(),  # browser mic audio in
            # VAD sits *before* STT: on "user stopped", Deepgram's service
            # sends a Finalize request that flushes the last words immediately
            # instead of waiting for Deepgram's own endpointing.
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            aggregators.user(),  # decides turn end; adds user text to context
            llm,
            # Retry a failed response once, then speak a fallback line.
            LLMErrorRecovery(llm, context),
            # "One moment." if the LLM is slow; never stored in context, and
            # not while the caller has asked us to hold.
            SlowResponseFiller(FILLER_DELAY_SECS, skip_if=lambda: pacing.on_hold),
            tts,
            transport.output(),  # bot audio back to the browser
            # After output, so the context records only what was actually
            # played. If the user interrupts, the unspoken rest is dropped.
            aggregators.assistant(),
        ]
    )

    latency = TurnLatencyObserver()
    worker = PipelineWorker(
        pipeline,
        # enable_metrics adds Pipecat's own per-service TTFB numbers to the
        # debug log, a useful cross-check for our TurnLatencyObserver.
        params=PipelineParams(enable_metrics=True),
        observers=[latency],
    )

    # Live latency readout: send each turn's timings to the browser as an
    # RTVI server message (text stays out; the client has its own transcript).
    @latency.event_handler("on_turn_metrics")
    async def on_turn_metrics(observer: TurnLatencyObserver, metrics: TurnMetrics) -> None:
        timings = {k: v for k, v in asdict(metrics).items() if k not in ("user_text", "bot_text")}
        await worker.rtvi.send_server_message({"type": "turn_metrics", **timings})

    # Hard cap on call length (public demo guardrail): wrap up, then end.
    timer = (
        CallTimer(max_call_secs, worker.queue_frames, lambda: runner.cancel())
        if max_call_secs
        else None
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport: BaseTransport, client: object) -> None:
        logger.info("Client connected")

    # Speak first. "client ready" (sent by the browser client over RTVI) comes
    # after "connected" and means it's ready to play audio, so the start of
    # the greeting isn't cut off.
    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor) -> None:
        context.add_message({"role": "developer", "content": GREETING_INSTRUCTION})
        await worker.queue_frames([LLMRunFrame()])
        if timer:
            timer.start()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: object) -> None:
        logger.info("Client disconnected")
        await runner.cancel()

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if timer:
            await timer.stop()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point the Pipecat runner calls once per browser session."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    # The runner stores the parsed CLI args (including our --brief) here.
    cli_args = runner_args.cli_args
    brief, label = resolve_brief(getattr(cli_args, "brief", None) if cli_args else None)
    logger.info(f"Using brief: {label} ({len(brief):,} chars)")

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, brief, handle_sigint=runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    # Pipecat's main() accepts our own parser: it adds its flags (--port,
    # -t, ...) to it and puts the parsed result on runner_args.cli_args.
    parser = argparse.ArgumentParser(description="CALL-E voice sales agent")
    add_brief_argument(parser)

    # Fail fast with clear messages if .env or the brief is bad, rather than
    # starting the server and failing when the first browser connects.
    # (A help-less pre-parse, so -h still shows Pipecat's flags too.)
    pre_parser = argparse.ArgumentParser(add_help=False)
    add_brief_argument(pre_parser)
    pre_args, _ = pre_parser.parse_known_args()
    try:
        load_settings()
        _, brief_label = resolve_brief(pre_args.brief)
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from None
    except KnowledgeError as error:
        raise SystemExit(f"Brief error: {error}") from None
    logger.info(f"Brief: {brief_label}")
    main(parser)
