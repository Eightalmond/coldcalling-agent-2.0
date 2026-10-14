"""CALL-E step 3: an LLM sales persona.

    mic -> VAD -> Deepgram STT -> user context -> Groq LLM -> Deepgram TTS
        -> speaker -> assistant context

Run it (opens a browser client at http://localhost:7860/client):

    uv run python -m calle.bot

Why `python -m calle.bot` and not a script entry point: Pipecat's dev
runner finds your `bot()` function by looking at the `__main__` module.
`-m` makes this file `__main__`; a console-script wrapper would not.
"""

import os

import certifi

# python.org builds of Python on macOS ship without trusted root certs, so
# Deepgram's websocket fails with CERTIFICATE_VERIFY_FAILED. Point OpenSSL at
# certifi's CA bundle. This must run before the pipecat imports below:
# aiohttp (used by pipecat) builds its SSL context once, at import time.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from calle.config import ConfigError, load_settings
from calle.log_redaction import install_log_redaction
from calle.metrics import TurnLatencyObserver
from calle.prompts import GREETING_INSTRUCTION, build_system_prompt

# Aura-2 "Thalia": clear, upbeat American English. A good fit for a
# friendly sales rep. Other voices: https://developers.deepgram.com/docs/tts-models
TTS_VOICE = "aura-2-thalia-en"
STT_MODEL = "nova-3-general"

# gpt-oss-20b is a reasoning model: it "thinks" before its first spoken word.
# "low" is the lowest effort Groq accepts for gpt-oss (there is no "none").
# The reasoning text arrives in a separate field that Pipecat doesn't speak.
LLM_MODEL = "openai/gpt-oss-20b"
LLM_REASONING_EFFORT = "low"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    settings = load_settings()
    system_prompt = build_system_prompt()
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
    # second Silero model. With default params the aggregator ends a turn with
    # the bundled Smart Turn v3 model and interrupts the bot on user speech.
    aggregators = LLMContextAggregatorPair(context)

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
            tts,
            transport.output(),  # bot audio back to the browser
            # After output, so the context records only what was actually
            # played. If the user interrupts, the unspoken rest is dropped.
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        # enable_metrics adds Pipecat's own per-service TTFB numbers to the
        # debug log, a useful cross-check for our TurnLatencyObserver.
        params=PipelineParams(enable_metrics=True),
        observers=[TurnLatencyObserver()],
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: object) -> None:
        logger.info("Client disconnected")
        await runner.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point the Pipecat runner calls once per browser session."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    # Fail fast with a clear message if .env is incomplete, rather than
    # starting the server and failing when the first browser connects.
    try:
        load_settings()
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from None
    main()
