"""Scrub secrets and the system prompt from every log line.

Pipecat's DEBUG logs print the LLM's full system instruction (once at
startup and whenever it changes). That's long, noisy, and in later steps will
contain user-provided product material, so we replace it with a short
placeholder. API keys are replaced too, as a safety net.

A loguru *patcher* runs on every record before any sink formats it. It is
global and survives the Pipecat runner's `logger.remove()` / `logger.add()`
when it sets the log level.
"""

from collections.abc import Iterable

from loguru import logger


def install_log_redaction(*, secrets: Iterable[str], system_prompt: str) -> None:
    """Patch all future log records: keys -> ***, system prompt -> placeholder."""
    secret_values = tuple(secret for secret in secrets if secret)
    prompt_placeholder = f"<system prompt, {len(system_prompt)} chars>"

    def redact(record: dict) -> None:
        message: str = record["message"]
        if system_prompt and system_prompt in message:
            message = message.replace(system_prompt, prompt_placeholder)
        for secret in secret_values:
            message = message.replace(secret, "***")
        record["message"] = message

    logger.configure(patcher=redact)
