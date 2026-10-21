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
  - `prompts.py` — `SYSTEM_PROMPT` template (`{PRODUCT_BRIEF}` placeholder, brief wrapped
    in `<product_brief>` as reference data), greeting instruction, `DEMO_BRIEF` fixture
    (fictional Quillo by Brightfern Labs, in the standard 7-heading format)
  - `knowledge.py` — Step 4: load material, optional Tavily web context, condense into a
    validated brief with Groq (`build_brief`), `read_brief_file` for the bot
  - `metrics.py` — `TurnLatencyObserver`, per-turn latency logging (reused in later steps)
  - `log_redaction.py` — loguru patcher that hides API keys and the system prompt in logs
  - `llm_recovery.py` — `LLMErrorRecovery`: retry a failed LLM response once, then a spoken fallback
  - `filler.py` — `SlowResponseFiller`: "One moment." if no LLM text within `FILLER_DELAY_SECS`
    (skipped while the caller asked to hold)
  - `turn_taking.py` — Step 5: `InterruptPhraseStartStrategy` (backchannel-proof barge-in)
    and `CallPacing` ("hold on" handling, idle check-ins, goodbye)
  - `echo.py` — step-2 `EchoProcessor`; no longer in the pipeline, kept for reference
- `scripts/` — runnable utilities (`smoke_test.py` checks every API key;
  `make_brief.py` builds a brief from material)
- `examples/quillo_material.md` — fake product sheet for testing, with deliberate gaps
  (no discount info, no SOC 2 / compliance, no SLA, no contract terms)
- `tests/` — offline pytest suite (fake LLM, simulated frames; no keys, no network)
- `briefs/` — generated briefs (git-ignored: may contain client material)
- `.env.example` — required + optional keys; real keys live in `.env` (git-ignored)

## Plan
1. **Setup** — uv project, dependencies, config loading, smoke test ✅
2. **STT + TTS echo bot** — hear the user, say it back ✅
3. **LLM sales persona** — Groq-powered salesperson ✅
4. **Product knowledge** — the user provides material (pasted text or uploaded
   `.txt`/`.md`/`.pdf`); it's condensed into a structured product brief that is
   injected into the system prompt. No RAG for the MVP. Tavily web search is an
   optional enrichment toggle, available only when `TAVILY_API_KEY` is set.
   ✅ live-tested 2026-10-21 (backend + CLI only; upload UI is Step 6)
5. **Interruptions / turn-taking** — barge-in, natural turn ends 🚧 *built, pending live test*
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
- Tests (offline): `uv run pytest`
- Build a brief: `uv run python scripts/make_brief.py --name Quillo --company "Brightfern Labs"
  --description "AI meeting notes" --file examples/quillo_material.md [--web]`
  (or `--text "..."` instead of `--file`; `--out PATH` to override `briefs/<slug>.md`)
- Run with a brief: `uv run python -m calle.bot --brief briefs/quillo.md`
  (or `CALLE_BRIEF=briefs/quillo.md`; flag wins; neither = built-in `DEMO_BRIEF`)
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
  turn is over (stop strategy: bundled Smart Turn v3.2 ONNX model + a final
  transcript), broadcasts `UserStoppedSpeakingFrame`, appends the user message to the
  `LLMContext` and pushes an `LLMContextFrame`. When a user turn *starts* it broadcasts
  an `InterruptionFrame` (barge-in); our start strategy decides when that is (Step 5).
  It gets no `vad_analyzer`: it reuses the VAD frames from our `VADProcessor` instead
  of running Silero twice. Every frame reaches the stop strategies whether or not a
  turn is active, so Smart Turn hears all the audio.
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
  DEBUG "VAD blip" line. A turn started by a transcript alone (no VAD stop of its
  own) is timed from "turn end" instead, and the log line says so.

## Step 5: turn-taking
- **Backchannel-proof barge-in.** Pipecat's default start strategies
  (`VADUserTurnStartStrategy` + `TranscriptionUserTurnStartStrategy`) start a turn, and
  so interrupt Alex, on any detected voice, including "mm-hmm" or a cough. Replaced by
  `InterruptPhraseStartStrategy(min_words=INTERRUPT_MIN_WORDS)`, a subclass of Pipecat's
  `MinWordsUserTurnStartStrategy`: while the bot speaks, a turn needs 3 words (counted
  on interim transcripts); when it's silent, 1 word. Utterances that *start* with an
  `INTERRUPT_PHRASES` entry ("wait", "stop", "hold on", "sorry", "no", ...) cut in
  immediately. Short backchannels are cleared via `trigger_reset_aggregation()`, so they
  don't leak into the next user message. Trade-off: barge-in now waits for the first
  interim transcript (a few hundred ms) instead of VAD.
- **Hold.** `CallPacing` reads the finished user turn (`on_user_turn_stopped`); a short
  utterance (<= 6 words) matching "hold on / hang on / one second / give me a moment /
  bear with me" sets `on_hold`, pushes `UserIdleTimeoutUpdateFrame(HOLD_IDLE_SECS=30)`,
  and suppresses the filler. The LLM still says the acknowledgement (prompt rule). The
  next user turn leaves hold and restores the 10 s timeout.
- **Idle check-in.** `LLMUserAggregatorParams(user_idle_timeout=IDLE_CHECKIN_SECS=10)`;
  Pipecat starts the timer when the bot stops speaking. On `on_user_turn_idle`,
  `CallPacing` speaks "Are you still there?" (or a hold variant), up to
  `MAX_IDLE_CHECKINS=2`, then `GOODBYE_TEXT` followed by `EndFrame` (queued behind the
  goodbye audio, so it plays fully). Check-ins are stored in context (the caller heard
  them). Any user turn resets the count.
- Tests: `tests/test_turn_taking.py` runs Pipecat's real `LLMUserAggregator` with our
  start strategy (mm-hmm / 2 words don't interrupt, 3 words and "wait" do) plus
  phrase matching, `CallPacing` with a fake aggregator, and the filler's hold hook.
- Not done (deliberately out of scope for now): interrupted-reply note in context,
  merging split user turns, barge-in latency metric. See Open issues.

## Step 4: product knowledge
```
file/text -> load_material (.txt/.md/.pdf via pypdf; scanned PDF -> error)
  -> limit_material (MAX_MATERIAL_CHARS = 24,000; truncate + warning, never silent)
  -> [--web + TAVILY_API_KEY: Tavily basic search "<product> <company>", 3 results]
  -> build_brief: one Groq request (BRIEF_MODEL = gpt-oss-120b, reasoning "medium";
     may differ from the live model since it runs before the call)
  -> validate_brief (7 exact `##` headings, <= BRIEF_MAX_CHARS = 3,000); 1 retry, then error
  -> briefs/<slug>.md -> bot --brief -> build_system_prompt(brief)
```
- Brief headings: Overview, Key features, Pricing and plans, Ideal customer, Common
  objections and answers, Competitors or alternatives, Not covered.
- Condensing rules: facts only from the input; web-only facts end with "(web)"; never
  invent prices/numbers/integrations/claims; unanswered buyer topics (discounts,
  integrations, security/compliance, contract terms, free trial, support SLA) go under
  "Not covered"; plain spoken-style sentences.
- Prompt-injection defense: PRODUCT_INFO, MATERIAL and WEB_CONTEXT are wrapped in
  `<<<NAME_START nonce>>>` / `<<<NAME_END nonce>>>` with a fresh random nonce per request
  (a document can't forge the end marker), and the condensing model is told to treat them
  as data and ignore instructions inside. The live prompt also wraps the brief in
  `<product_brief>` and calls it reference data, not instructions.
- Live-call rules: "Not covered" or unanswered -> "not sure, the team will follow up",
  never guess; "(web)" facts phrased softly ("I believe...").
- Never logged: API keys, the material, the brief text. The bot logs only the brief's
  file name and length; `make_brief.py` prints only warnings, path and size.
- `_clean_reply` strips code fences and `**bold**`/`__bold__` from the model's brief (the
  prompt forbids bold, but gpt-oss-120b added it anyway on the first live run).
- Tests: `uv run pytest` (`test_knowledge.py`: each file type, scanned PDF, truncation
  warning, heading validation + retry, LLM-error retry, bold stripping, web block labeling
  and the no-key warning, injected instructions stay inside nonce delimiters;
  `test_metrics.py`: VAD-stop anchor, turn-end fallback, VAD blip, filler exclusion).

## Live test 2026-10-21 (Quillo brief from examples/quillo_material.md)
- Passed: pricing; discounts, SOC 2 and Salesforce all answered "not sure, the team will
  follow up"; languages; "are you a real person" -> AI; "hold on" -> "Sure, take your
  time."; barge-in stops the bot; a mid-sentence pause got Smart Turn INCOMPLETE and the
  bot did not cut in. Greeting via the `developer` role works on live Groq. No LLM errors.
- Latency: VAD stop -> first audio 0.9 to 1.45 s (~1.2 s typical); first token 540 to
  1080 ms after VAD stop; first token -> first audio 340 to 480 ms. Filler never fired.
- Fixed after the test: replies ran 33 to 50 words by chaining clauses with dashes
  (prompt now: one point, hard 30-word cap, no dash/semicolon chaining); the bot offered a
  demo not in the brief (prompt now: only next steps from the brief or a team follow-up);
  turns started by a transcript alone logged `n/a` timings (metrics now fall back to the
  turn-end anchor and label it); bold in generated briefs (now stripped in code).
- Not yet re-tested live after these fixes.

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
  turn-taking logic (e.g. not answering "hold on" at all). "hold on" worked live; "go on"
  hasn't been tested mid-explanation yet.
- **Split turns.** A mid-sentence pause can split one utterance into two user turns (the
  first LLM response is cancelled before any audio, so the caller doesn't hear it). The
  context then holds two consecutive user messages. Revisit in Step 5.
- **Reply length is prompt-enforced only.** If it keeps overrunning, add a processor that
  stops passing LLM text to TTS after the second sentence.

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
