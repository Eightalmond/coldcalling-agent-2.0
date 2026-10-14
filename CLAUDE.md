# CALL-E — project context

## Goal
Rebuild CALL-E, a real-time AI sales voice agent, as a clean portfolio project.
The 2024 version used Flask + Deepgram + GPT-3.5. This version uses a proper
streaming pipeline. The owner wants to understand every stage, so briefly
explain non-obvious choices when making them.

## Hard constraint: $0 to run
Everything must work on free tiers. **Ask before adding any paid service or
any dependency that needs a paid account.**

## Stack
- Python 3.12, managed with **uv** (`uv add`, `uv run`; no pip/requirements.txt)
- **Pipecat** (`pipecat-ai`) for orchestration. Extras installed:
  `deepgram`, `groq`, `silero` (VAD), `webrtc` (SmallWebRTCTransport), `runner` (dev runner + prebuilt client)
- **Deepgram** for STT and TTS
- **Groq** for the LLM
- **Tavily** (`tavily-python`), optional web-search enrichment (off when `TAVILY_API_KEY` is unset)
- Small web client (step 6)

## Layout
- `src/calle/` — the package
  - `config.py` — loads/validates `.env`
  - `bot.py` — pipeline wiring + Pipecat runner entry point (`bot()`); model/voice constants
  - `prompts.py` — `SYSTEM_PROMPT` template (`{PRODUCT_BRIEF}` placeholder), greeting
    instruction, fictional `DEMO_PRODUCT_BRIEF` (Quillo by Brightfern Labs)
  - `metrics.py` — `TurnLatencyObserver`, per-turn latency logging (reused in later steps)
  - `log_redaction.py` — loguru patcher that hides API keys and the system prompt in logs
  - `llm_recovery.py` — `LLMErrorRecovery`: retry a failed LLM response once, then a spoken fallback
  - `filler.py` — `SlowResponseFiller`: "One moment." if no LLM text within `FILLER_DELAY_SECS`
  - `echo.py` — step-2 `EchoProcessor`; no longer in the pipeline, kept for reference
- `scripts/` — runnable utilities (`smoke_test.py` checks every API key)
- `.env.example` — required + optional keys; real keys live in `.env` (git-ignored)

## Plan
1. **Setup** — uv project, dependencies, config loading, smoke test ✅
2. **STT + TTS echo bot** — hear the user, say it back ✅
3. **LLM sales persona** — Groq-powered salesperson 🚧 *in progress*
4. **Product knowledge** — the user provides material (pasted text or uploaded
   `.txt`/`.md`/`.pdf`); it's condensed into a structured product brief that is
   injected into the system prompt. No RAG for the MVP. Tavily web search is an
   optional enrichment toggle, available only when `TAVILY_API_KEY` is set.
5. **Interruptions / turn-taking** — barge-in, natural turn ends
6. **Web UI + guardrails** — call duration cap, rate limit, latency readout
7. **Polish** — README, architecture diagram, demo video

Work one step at a time; don't build ahead.

## Conventions
- Type hints on all functions.
- Small, single-purpose modules.
- No secrets in code, logs, or output. Read keys only via `calle.config.load_settings()`.
- Never create `.env` or read real keys; the owner manages those.
- Ask before adding paid services.

## Commands
- Install/sync: `uv sync`
- Smoke test: `uv run python scripts/smoke_test.py`
- Run the bot: `uv run python -m calle.bot`, then open http://localhost:7860 (allow the mic).
  Must be `-m calle.bot`: the Pipecat runner looks for `bot()` on the `__main__` module,
  so a `[project.scripts]` entry point won't work. Add `-v` for verbose logs.

## How the pipeline works
```
transport.input() -> VADProcessor(Silero) -> DeepgramSTTService
  -> user aggregator -> GroqLLMService -> LLMErrorRecovery -> SlowResponseFiller
  -> DeepgramTTSService -> transport.output() -> assistant aggregator
```

- Everything is a **frame** flowing downstream through processors. Audio in is
  `InputAudioRawFrame`; audio out is `TTSAudioRawFrame`.
- **VAD** passes audio through and adds `VADUserStarted/StoppedSpeakingFrame`
  (after `stop_secs` = 0.2 s of silence). It sits before STT because Deepgram's
  service sends a `Finalize` on "stopped", which flushes the last words fast.
- **STT** emits `InterimTranscriptionFrame`s while you talk and `TranscriptionFrame`s
  for final text. Deepgram can emit several finals per utterance.
- **User aggregator** (`LLMContextAggregatorPair(context).user()`) decides when the
  turn is over (default stop strategy: bundled Smart Turn v3.2 ONNX model + a final
  transcript), broadcasts `UserStoppedSpeakingFrame`, appends the user message to the
  `LLMContext` and pushes an `LLMContextFrame`. On user speech it broadcasts an
  `InterruptionFrame`, so basic barge-in already works. It gets no `vad_analyzer`:
  it reuses the VAD frames from our `VADProcessor` instead of running Silero twice.
- **LLM** (`GroqLLMService`) streams `LLMFullResponseStartFrame`, `LLMTextFrame`s,
  `LLMFullResponseEndFrame`. The system prompt is passed as
  `Settings(system_instruction=...)`, not as a context message.
- **TTS** aggregates LLM tokens into sentences, then emits `TTSAudioRawFrame`s; the
  output transport streams them to the browser.
- **Assistant aggregator** sits *after* the output so the context records only what
  was actually spoken (an interrupted reply is truncated).
- **Greeting**: on RTVI `on_client_ready` (client ready to play audio, later than
  `on_client_connected`), add a `developer` message and queue an `LLMRunFrame`.
- **Observers** (`TurnLatencyObserver`) watch frames from outside the pipeline, so
  measuring never adds latency. Per turn it logs user text, bot text, VAD wait, STT
  delay, turn end, VAD stop -> first LLM token, first token -> first audio, and
  total (VAD stop -> first audio). Turns with an empty transcript log only a
  DEBUG "VAD blip" line.

## LLM model choice
`openai/gpt-oss-20b` with `reasoning_effort="low"` (lowest Groq allows for gpt-oss;
reasoning text is returned separately and never spoken, but it still delays the
first token). Groq's non-reasoning Llama models are marked "Enterprise"/contact-sales;
`qwen/qwen3.8-27b` (preview) accepts `reasoning_effort="none"`. Compare on the
"VAD stop -> first token" metric before switching.

## Step 3 baseline (measured on real calls)
- VAD stop -> first audio: about 1.1 to 1.3 s on normal turns.
- Groq TTFB: about 0.5 s (Pipecat's own LLM TTFB metric, `-v` logs).
- Smart Turn silence fallback (`SMART_TURN_STOP_SECS`, i.e. `SmartTurnParams.stop_secs`)
  lowered from Pipecat's default 3 s to 1.5 s; re-measure "turn end" after this change.

## Known issue: gpt-oss "Tool choice is none, but model called a tool"
- **Symptom:** on some turns (seen on a pricing question) gpt-oss-20b emits a tool call
  although no tools are defined. Groq rejects it with HTTP 400 (`tool_use_failed`),
  Pipecat logs a non-fatal ErrorFrame, and the bot went silent ~13 s until the caller
  said "Hello?". Pipecat also files every HTTP 400 as a *permanent* `INVALID_REQUEST`
  and flags the LLM unusable (harmless with the default `ProcessorUnusablePolicy.CONTINUE`,
  but misleading).
- **Fix:**
  1. Prompt rule: no tools; never try to call a function or look anything up.
  2. `LLMErrorRecovery` (after the LLM) listens to `llm`'s `on_error` event, re-marks the
     LLM usable, and when the failed response's `LLMFullResponseEndFrame` passes, either
     retries once by pushing the same `LLMContextFrame` upstream (what the assistant
     aggregator does for `LLMRunFrame`) or, on a second failure, speaks
     "Sorry, I lost my train of thought. Could you say that again?". A failed attempt
     adds nothing to context (empty aggregation), a retry reuses the same context (no
     duplicate user message), and the fallback is stored as an assistant message on
     purpose (the caller heard it). No retry if part of the reply was already spoken.
  3. `SlowResponseFiller` speaks a varied filler after `FILLER_DELAY_SECS` (1.5 s) with
     no LLM text; cancelled by LLM text, user speech, or interruption; one per user turn;
     never on the greeting. Sent with `append_to_context=False`, so it is not stored in
     context. Deepgram TTS pauses its input while the filler plays, so an answer that
     arrives mid-filler starts right after it.
  4. Metrics: each failure logs a WARNING `LLM error (retrying|speaking fallback) |
     errors=N retries=N fallbacks=N | <error text>` (truncated, never the prompt). Turn
     logs show `filler at X ms`; filler audio is excluded from first-audio timing.
- **If errors stay frequent:** try `qwen/qwen3.8-27b` with `reasoning_effort="none"`
  (non-reasoning, so also lower first-token latency; it's a preview model). Compare
  the `errors=` counter and "VAD stop -> first token" before switching.

## Open issues
- **Interrupted replies in context (sentence granularity).** Deepgram TTS has no word
  timestamps (Aura's websocket only sends audio, Metadata, Flushed, Cleared, Warning), so
  Pipecat emits one `TTSTextFrame` per sentence, queued *behind* that sentence's audio in
  the output transport. The assistant aggregator (after `transport.output()`) only sees
  sentences whose audio was fully played; on interruption the output queue is cleared and
  `_trigger_assistant_turn_stopped(interrupted=True)` writes what arrived so far. Result:
  the context holds fully spoken sentences only; a half-spoken sentence is dropped
  entirely, so the LLM doesn't know it said part of it. Not changed yet.
- **`go on` / `hold on` handling** is prompt-only for now; Step 5 may need real
  turn-taking logic (e.g. not answering "hold on" at all).
- **Greeting `developer` role** on Groq is unverified against Groq's docs (Pipecat
  treats Groq as supporting it); switch to `user` if it ever returns HTTP 400.

## Gotchas
- Since Pipecat 1.3, `PipelineTask`/`PipelineRunner` are deprecated aliases; use `PipelineWorker`/`WorkerRunner`,
  and service options go through `Service.Settings(...)`, not constructor kwargs.
  Check the installed source under `.venv/lib/python3.12/site-packages/pipecat/` before trusting memory.
- Importing `pipecat.runner.run` calls `load_dotenv(override=True)`, so when running the bot,
  `.env` values beat shell env vars. Set `PYTHON_DOTENV_DISABLED=1` to test with env vars only.
- macOS python.org Python has no trusted root certs, so websockets fail with
  `CERTIFICATE_VERIFY_FAILED`. `bot.py` sets `SSL_CERT_FILE` to `certifi.where()` (unless
  already set) at the very top, before pipecat/aiohttp imports build their SSL context.
- Pipecat's DEBUG logs print the LLM `system_instruction` and, on every LLM call, the
  full context messages. Keep the system prompt out of the context and keep
  `install_log_redaction()` installed so it never appears in logs.
