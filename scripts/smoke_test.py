"""Smoke test: one tiny real call to each external service.

Run with:  uv run python scripts/smoke_test.py

Prints PASS/FAIL per service and exits non-zero if anything fails.
API keys are never printed; they're scrubbed from any error text too.
"""

import sys
from collections.abc import Callable

from deepgram import DeepgramClient
from groq import Groq
from tavily import TavilyClient

from calle.config import ConfigError, Settings, load_settings

# Cheapest/fastest options on each free tier.
DEEPGRAM_TTS_MODEL = "aura-2-thalia-en"
DEEPGRAM_STT_MODEL = "nova-3"
GROQ_MODEL = "openai/gpt-oss-20b"


def check_deepgram(settings: Settings) -> str:
    """Round trip: synthesize a phrase (TTS), then transcribe it back (STT).

    One key covers both directions, and feeding TTS output into STT checks
    both without needing a sample audio file in the repo.
    """
    client = DeepgramClient(api_key=settings.deepgram_api_key)

    # generate() streams audio in chunks; join them into one WAV blob.
    audio = b"".join(
        client.speak.v1.audio.generate(
            text="Hello from CALL-E.",
            model=DEEPGRAM_TTS_MODEL,
            encoding="linear16",
            container="wav",
        )
    )
    response = client.listen.v1.media.transcribe_file(
        request=audio,
        model=DEEPGRAM_STT_MODEL,
    )
    transcript = response.results.channels[0].alternatives[0].transcript
    if not transcript:
        raise RuntimeError("TTS worked but STT returned an empty transcript")
    return f"TTS {len(audio):,} bytes -> STT heard {transcript!r}"


def check_groq(settings: Settings) -> str:
    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Reply with just the word: pong"}],
        # gpt-oss is a reasoning model; keep reasoning short so the
        # token budget goes to the answer.
        reasoning_effort="low",
        max_completion_tokens=64,
    )
    reply = (completion.choices[0].message.content or "").strip()
    return f"{GROQ_MODEL} replied {reply!r}"


def check_tavily(settings: Settings) -> str:
    client = TavilyClient(api_key=settings.tavily_api_key)
    # "basic" depth costs 1 credit (vs 2 for "advanced") on the free plan.
    result = client.search("Pipecat voice AI framework", search_depth="basic", max_results=1)
    hits = result.get("results", [])
    if not hits:
        raise RuntimeError("search succeeded but returned no results")
    return f"top hit: {hits[0]['url']}"


CHECKS: dict[str, Callable[[Settings], str]] = {
    "Deepgram": check_deepgram,
    "Groq": check_groq,
    "Tavily": check_tavily,
}


def describe(error: Exception) -> str:
    """One readable line per error.

    SDK HTTP errors (Deepgram, Groq) carry `status_code` and `body`; their
    default str() dumps every response header, which buries the cause.
    """
    status, body = getattr(error, "status_code", None), getattr(error, "body", None)
    if status is not None and body is not None:
        return f"{type(error).__name__} (HTTP {status}): {body}"
    return f"{type(error).__name__}: {error}"


def redact(text: str, secrets: tuple[str, ...]) -> str:
    """Replace any secret that slipped into an error message with ***."""
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Config error: {error}")
        return 1

    failures = 0
    for name, check in CHECKS.items():
        try:
            detail = check(settings)
            print(f"PASS  {name:<9} {detail}")
        except Exception as error:  # report every service, don't stop at the first failure
            failures += 1
            message = redact(describe(error), settings.secrets())
            print(f"FAIL  {name:<9} {message}")

    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} services OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
