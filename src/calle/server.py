"""CALL-E web server: the API for the web client, plus the WebRTC signalling.

Run locally:
    uv run python -m calle.server          # http://localhost:7860

This replaces Pipecat's dev runner for the web UI, because we need our own
routes (brief upload, call guardrails) and control over who may connect.

Flow:
  1. POST /api/briefs   material (upload or text) -> generated brief (rate limited)
  2. POST /api/calls    brief (possibly edited) -> validated, limits checked,
                        one-time call_id
  3. POST /api/offer    WebRTC offer carrying {call_id}; unknown/used ids are
                        rejected before any WebRTC resources are created
  4. The bot runs in the background for that connection, capped at
     CALL_MAX_SECS.

If client/dist exists (`npm run build` in client/), it's served at "/", so
one process serves both UI and API locally. In production the UI can live
on a static host instead; set CALLE_ALLOWED_ORIGINS for CORS.

Env (all optional):
  CALLE_ALLOWED_ORIGINS  comma-separated origins allowed to call the API
  CALLE_TRUST_PROXY=1    use X-Forwarded-For for the visitor IP (behind a proxy)
  CALLE_ICE_SERVERS      JSON list of RTCIceServer dicts (STUN/TURN), shared
                         with the browser via /api/config
"""

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from calle import bot  # noqa: F401  (sets SSL_CERT_FILE before pipecat imports)
from calle.config import ConfigError, Settings, load_settings
from calle.guardrails import (
    BRIEFS_PER_VISITOR_PER_HOUR,
    CALL_MAX_SECS,
    CALLS_PER_DAY,
    CALLS_PER_VISITOR_PER_HOUR,
    HOUR,
    CallRegistry,
    LimitExceeded,
    SlidingWindowLimiter,
)
from calle.knowledge import (
    BRIEF_MAX_CHARS,
    SUPPORTED_SUFFIXES,
    BriefLLM,
    KnowledgeError,
    ProductInput,
    WebSearch,
    build_brief,
    load_material,
    validate_brief,
)
from calle.llm_models import LLM_ENV_VAR, resolve_llm
from calle.prompts import DEMO_BRIEF

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:7860"
CLIENT_DIST = Path(__file__).resolve().parents[2] / "client" / "dist"

# Runs one call. Injected so tests don't start a real voice pipeline.
CallRunner = Callable[[SmallWebRTCConnection, str], Awaitable[None]]


class CallRequest(BaseModel):
    brief: str


async def run_call(connection: SmallWebRTCConnection, brief: str) -> None:
    transport = SmallWebRTCTransport(
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        webrtc_connection=connection,
    )
    await bot.run_bot(transport, brief, max_call_secs=CALL_MAX_SECS)


def _ice_servers_from_env() -> list[dict[str, Any]]:
    raw = os.getenv("CALLE_ICE_SERVERS", "").strip()
    if not raw:
        return []
    servers = json.loads(raw)
    if not isinstance(servers, list):
        raise ConfigError("CALLE_ICE_SERVERS must be a JSON list of ICE server objects")
    return servers


def create_app(
    settings: Settings | None = None,
    *,
    brief_llm: BriefLLM | None = None,
    web_search: WebSearch | None = None,
    call_runner: CallRunner = run_call,
    registry: CallRegistry | None = None,
    brief_limiter: SlidingWindowLimiter | None = None,
    webrtc: SmallWebRTCRequestHandler | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    registry = registry or CallRegistry()
    brief_limiter = brief_limiter or SlidingWindowLimiter(BRIEFS_PER_VISITOR_PER_HOUR, HOUR)
    ice_servers = _ice_servers_from_env()
    webrtc = webrtc or SmallWebRTCRequestHandler(
        ice_servers=[IceServer(**s) for s in ice_servers] or None
    )
    trust_proxy = os.getenv("CALLE_TRUST_PROXY") == "1"
    # Peer connections we accepted, so renegotiation/ICE requests for them pass.
    accepted_pcs: set[str] = set()
    call_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await webrtc.close()

    app = FastAPI(title="CALL-E", lifespan=lifespan)
    origins = os.getenv("CALLE_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    def visitor_of(request: Request) -> str:
        if trust_proxy and (forwarded := request.headers.get("x-forwarded-for")):
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def too_many(error: LimitExceeded) -> JSONResponse:
        retry = max(1, round(error.retry_after))
        return JSONResponse(
            {"detail": str(error), "retry_after": retry},
            status_code=429,
            headers={"Retry-After": str(retry)},
        )

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "web_search_available": bool(settings.tavily_api_key),
            "call_max_secs": CALL_MAX_SECS,
            "calls_per_hour": CALLS_PER_VISITOR_PER_HOUR,
            "calls_per_day": CALLS_PER_DAY,
            "brief_max_chars": BRIEF_MAX_CHARS,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "supported_files": list(SUPPORTED_SUFFIXES),
            "ice_servers": ice_servers,
        }

    @app.get("/api/demo-brief")
    async def demo_brief() -> dict[str, str]:
        return {"brief": DEMO_BRIEF.strip()}

    @app.post("/api/briefs", response_model=None)
    async def create_brief(
        request: Request,
        product_name: str = Form(...),
        company_name: str = Form(...),
        description: str = Form(""),
        text: str = Form(""),
        use_web: bool = Form(False),
        file: UploadFile | None = File(None),
    ) -> dict[str, Any] | JSONResponse:
        visitor = visitor_of(request)
        wait = brief_limiter.retry_after(visitor)
        if wait:
            return too_many(
                LimitExceeded(
                    f"You've generated {BRIEFS_PER_VISITOR_PER_HOUR} briefs this hour. "
                    "Try again later, or edit the current brief.",
                    wait,
                )
            )
        material = await _read_material(file, text)
        product = ProductInput(
            product_name=product_name.strip(),
            company_name=company_name.strip(),
            description=description.strip(),
            material_text=material,
            use_web=use_web,
        )
        brief_limiter.record(visitor)
        try:
            # build_brief makes a blocking Groq request; keep the event loop
            # (and any live calls) responsive while it runs.
            result = await run_in_threadpool(
                build_brief, product, settings, llm=brief_llm, web_search=web_search
            )
        except KnowledgeError as error:
            raise HTTPException(422, str(error)) from error
        logger.info(f"Brief generated ({len(result.brief):,} chars, web={result.used_web})")
        return {"brief": result.brief, "warnings": result.warnings, "used_web": result.used_web}

    @app.post("/api/calls", response_model=None)
    async def create_call(request: Request, body: CallRequest) -> dict[str, Any] | JSONResponse:
        brief = body.brief.strip()
        problems = validate_brief(brief)
        if problems:
            return JSONResponse(
                {"detail": "The brief needs fixing before a call.", "problems": problems},
                status_code=422,
            )
        try:
            call = registry.create(brief, visitor_of(request))
        except LimitExceeded as error:
            return too_many(error)
        logger.info(f"Call {call.call_id[:8]} created ({len(brief):,}-char brief)")
        return {"call_id": call.call_id, "max_call_secs": CALL_MAX_SECS}

    @app.post("/api/offer")
    async def offer(raw: Request) -> dict[str, Any]:
        # SmallWebRTCRequest is a plain dataclass: FastAPI would silently drop
        # the `requestData` key the JS client sends. from_dict() maps it.
        try:
            request = SmallWebRTCRequest.from_dict(await raw.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "Invalid WebRTC offer.") from error
        if not (request.pc_id and request.pc_id in accepted_pcs):
            data = request.request_data if isinstance(request.request_data, dict) else {}
            call_id = str(data.get("call_id", ""))
            call = registry.claim(call_id) if call_id else None
            if call is None:
                raise HTTPException(403, "Unknown, expired or already used call id.")
        else:
            call = None  # renegotiation of a connection we already accepted

        async def on_connection(connection: SmallWebRTCConnection) -> None:
            assert call is not None
            accepted_pcs.add(connection.pc_id)
            task = asyncio.create_task(_run_and_finish(connection, call.call_id, call.brief))
            call_tasks.add(task)
            task.add_done_callback(call_tasks.discard)

        return await webrtc.handle_web_request(
            request=request, webrtc_connection_callback=on_connection
        )

    @app.patch("/api/offer")
    async def ice_candidate(request: SmallWebRTCPatchRequest) -> dict[str, str]:
        if request.pc_id not in accepted_pcs:
            raise HTTPException(404, "Unknown peer connection.")
        await webrtc.handle_patch_request(request)
        return {"status": "success"}

    async def _run_and_finish(connection: SmallWebRTCConnection, call_id: str, brief: str) -> None:
        logger.info(f"Call {call_id[:8]} connected")
        try:
            await call_runner(connection, brief)
        except Exception:
            logger.exception(f"Call {call_id[:8]} crashed")
        finally:
            registry.finish(call_id)
            accepted_pcs.discard(connection.pc_id)
            logger.info(f"Call {call_id[:8]} ended")

    if CLIENT_DIST.is_dir():
        app.mount("/", StaticFiles(directory=CLIENT_DIST, html=True), name="client")

    return app


async def _read_material(file: UploadFile | None, text: str) -> str:
    """Return material text from an upload or pasted text. Raises HTTPException."""
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                422, f"Unsupported file type {suffix or '(none)'!r}. Use {', '.join(SUPPORTED_SUFFIXES)}."
            )
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        # load_material reads from a path (pypdf needs a file); use a temp
        # file with the same suffix and delete it right after.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"material{suffix}"
            path.write_bytes(data)
            try:
                return await run_in_threadpool(load_material, path)
            except KnowledgeError as error:
                raise HTTPException(422, str(error)) from error
    if text.strip():
        return text
    raise HTTPException(422, "Provide product material: upload a file or paste text.")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="CALL-E web server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    try:
        app = create_app()
        preset = resolve_llm()  # fail fast on a bad CALLE_LLM_MODEL
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from None
    logger.info(f"LLM: {preset.model} (set {LLM_ENV_VAR} to change)")
    if not CLIENT_DIST.is_dir():
        logger.warning("client/dist not found: API only. Run `npm run build` in client/ for the UI.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
