"""Load and validate CALL-E's settings from the environment.

Keys come from a local `.env` file (see `.env.example`) or from real
environment variables. Real env vars win, so a deployment can override
`.env` without editing it.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REQUIRED_KEYS: tuple[str, ...] = (
    "DEEPGRAM_API_KEY",
    "GROQ_API_KEY",
)

# Features that switch off cleanly when their key is absent.
# TAVILY_API_KEY enables optional web-search enrichment of the product brief.
OPTIONAL_KEYS: tuple[str, ...] = ("TAVILY_API_KEY",)


class ConfigError(RuntimeError):
    """Raised when required settings are missing."""


@dataclass(frozen=True)
class Settings:
    deepgram_api_key: str
    groq_api_key: str
    tavily_api_key: str | None = None  # None = web enrichment disabled

    def __repr__(self) -> str:
        # The dataclass default repr would print every key in full.
        # Hide them so a stray print() or traceback can't leak secrets.
        tavily = "***" if self.tavily_api_key else "None"
        return f"Settings(deepgram_api_key=***, groq_api_key=***, tavily_api_key={tavily})"

    def secrets(self) -> tuple[str, ...]:
        """Every secret value that is set, so callers can redact them from output."""
        keys = (self.deepgram_api_key, self.groq_api_key, self.tavily_api_key)
        return tuple(key for key in keys if key)


def load_settings() -> Settings:
    """Read `.env`, check that every required key is set, and return Settings.

    Raises ConfigError listing *all* missing required keys at once, so you
    can fix them in one pass instead of one failure at a time. Optional keys
    that are missing or blank come back as None.
    """
    # override=False: an existing env var takes priority over the .env file.
    load_dotenv(override=False)

    values = {key: os.getenv(key, "").strip() for key in REQUIRED_KEYS}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ConfigError(
            "Missing required settings: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill them in."
        )

    return Settings(
        deepgram_api_key=values["DEEPGRAM_API_KEY"],
        groq_api_key=values["GROQ_API_KEY"],
        # `or None` turns a blank `TAVILY_API_KEY=` line into "not set".
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
    )
