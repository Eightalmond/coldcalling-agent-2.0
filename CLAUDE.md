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
- **Tavily** (`tavily-python`) for product research
- Small web client (step 6)

## Layout
- `src/calle/` — the package
  - `config.py` — loads/validates `.env`
  - `bot.py` — pipeline wiring + Pipecat runner entry point (`bot()`)
  - `echo.py` — `EchoProcessor`, the step-2 stand-in for the LLM
  - `metrics.py` — `TurnLatencyObserver`, per-turn latency logging (reused in later steps)
- `scripts/` — runnable utilities (`smoke_test.py` checks every API key)
- `.env.example` — required keys; real keys live in `.env` (git-ignored)

## Plan
1. **Setup** — uv project, dependencies, config loading, smoke test ✅
2. **STT + TTS echo bot** — hear the user, say it back ✅
3. **LLM sales persona** — Groq-powered salesperson
4. **Product research module** — Tavily lookups feeding the pitch
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
`transport.input() -> VADProcessor(Silero) -> DeepgramSTTService -> EchoProcessor -> DeepgramTTSService -> transport.output()`

- Everything is a **frame** flowing downstream through processors. Audio in is
  `InputAudioRawFrame`; audio out is `TTSAudioRawFrame`.
- **VAD** passes audio through and adds `VADUserStarted/StoppedSpeakingFrame`
  (after `stop_secs` = 0.2 s of silence). It sits before STT because Deepgram's
  service sends a `Finalize` on "stopped", which flushes the last words fast.
- **STT** emits `InterimTranscriptionFrame`s while you talk and `TranscriptionFrame`s
  for final text. Deepgram can emit several finals per utterance.
- **EchoProcessor** buffers finals and, once the user has stopped, pushes one
  `TTSSpeakFrame`. Step 3 replaces it with the LLM.
- **TTS** turns text into audio frames; the output transport streams them to the browser.
- **Observers** (`TurnLatencyObserver`) watch frames from outside the pipeline, so
  measuring never adds latency. Per turn it logs: user text, bot text, VAD wait,
  STT delay (signed; negative = transcript was ready before the VAD stop) and
  response time (VAD stop -> first TTS audio frame).

## Gotchas
- Since Pipecat 1.3, `PipelineTask`/`PipelineRunner` are deprecated aliases; use `PipelineWorker`/`WorkerRunner`,
  and service options go through `Service.Settings(...)`, not constructor kwargs.
  Check the installed source under `.venv/lib/python3.12/site-packages/pipecat/` before trusting memory.
- Importing `pipecat.runner.run` calls `load_dotenv(override=True)`, so when running the bot,
  `.env` values beat shell env vars. Set `PYTHON_DOTENV_DISABLED=1` to test with env vars only.
