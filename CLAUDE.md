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
- `src/calle/` — the package (`config.py` loads/validates `.env`)
- `scripts/` — runnable utilities (`smoke_test.py` checks every API key)
- `.env.example` — required keys; real keys live in `.env` (git-ignored)

## Plan
1. **Setup** — uv project, dependencies, config loading, smoke test ✅
2. **STT + TTS echo bot** — hear the user, say it back
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
