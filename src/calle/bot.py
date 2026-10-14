"""CALL-E step 2: an echo bot.

    mic -> VAD -> Deepgram STT -> EchoProcessor -> Deepgram TTS -> speaker

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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from calle.config import ConfigError, load_settings
from calle.echo import EchoProcessor
from calle.metrics import TurnLatencyObserver

# Aura-2 "Thalia": clear, upbeat American English. A good fit for a
# friendly sales rep. Other voices: https://developers.deepgram.com/docs/tts-models
TTS_VOICE = "aura-2-thalia-en"
STT_MODEL = "nova-3-general"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    settings = load_settings()

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(model=STT_MODEL),
    )
    tts = DeepgramTTSService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramTTSService.Settings(voice=TTS_VOICE),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # browser mic audio in
            # VAD sits *before* STT: on "user stopped", Deepgram's service
            # sends a Finalize request that flushes the last words immediately
            # instead of waiting for Deepgram's own endpointing.
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            EchoProcessor(),
            tts,
            transport.output(),  # bot audio back to the browser
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
        logger.info("Client connected. Say something!")

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
