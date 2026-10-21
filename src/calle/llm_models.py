"""Which Groq model runs the live call, chosen with the CALLE_LLM_MODEL env var.

Presets bundle the settings a model needs, so switching is one variable:

    CALLE_LLM_MODEL=qwen uv run python -m calle.server

The same presets drive scripts/eval_persona.py, so what we measure is what
we run. See docs/evals/ for how the models compared.
"""

import os
from dataclasses import dataclass

from calle.config import ConfigError

LLM_ENV_VAR = "CALLE_LLM_MODEL"


@dataclass(frozen=True)
class LLMPreset:
    model: str
    # gpt-oss is a reasoning model: "low" is the lowest Groq accepts, and its
    # reasoning (never spoken) still delays the first token. Qwen accepts
    # "none", which turns reasoning off.
    reasoning_effort: str
    # Qwen's chat template on Groq rejects the "developer" role (HTTP 400), so
    # Pipecat must send our greeting instruction as a "user" message instead.
    supports_developer_role: bool
    note: str


LLM_PRESETS: dict[str, LLMPreset] = {
    "gpt-oss": LLMPreset(
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
        supports_developer_role=True,
        note="production model on Groq (default)",
    ),
    "qwen": LLMPreset(
        model="qwen/qwen3.8-27b",
        reasoning_effort="none",
        supports_developer_role=False,
        note="preview model on Groq: no reasoning, faster first token, may be withdrawn",
    ),
}
DEFAULT_LLM = "gpt-oss"


def resolve_llm(name: str | None = None) -> LLMPreset:
    """Return the preset for `name` (or $CALLE_LLM_MODEL, or the default).

    Accepts a preset key ("qwen") or a full model id ("qwen/qwen3.8-27b").
    Raises ConfigError for anything else, listing the options.
    """
    choice = (name or os.getenv(LLM_ENV_VAR, "") or DEFAULT_LLM).strip()
    if choice in LLM_PRESETS:
        return LLM_PRESETS[choice]
    for preset in LLM_PRESETS.values():
        if choice == preset.model:
            return preset
    options = ", ".join(f"{key} ({p.model})" for key, p in LLM_PRESETS.items())
    raise ConfigError(f"Unknown {LLM_ENV_VAR}={choice!r}. Choose one of: {options}")
