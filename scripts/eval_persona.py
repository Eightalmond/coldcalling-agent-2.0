"""Compare LLMs on the live-call persona: rule adherence + time to first token.

Replays tricky conversation moments (from real test calls) against each model
on Groq, using the real system prompt and the demo brief, and checks every
reply automatically:

  - length: at most 2 sentences / ~30 words, at most one question
  - style: no dashes or semicolons chaining clauses
  - honesty: no numbers or claims missing from the brief, no invented website,
    no promises to do things itself ("I'll send / arrange / set up ...")
  - behaviour per scenario: "not sure" when the brief doesn't say, a needs
    question on the first "no", a goodbye on the second "no" or an opt-out

Run:  uv run python scripts/eval_persona.py [--runs 2] [--out docs/evals/<name>.md]

Costs a few dozen small Groq requests on the free tier. Prints replies and
results; never prints keys or the system prompt.
"""

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import groq

from calle.config import ConfigError, load_settings
from calle.llm_models import LLM_PRESETS
from calle.prompts import DEMO_BRIEF, GREETING_INSTRUCTION, build_system_prompt

# The same presets the bot uses (calle.llm_models): model, reasoning effort,
# and whether the model accepts the "developer" role. For models that don't
# (Qwen on Groq), the greeting instruction is sent as a "user" message, the
# same conversion Pipecat does when supports_developer_role is False.
MODELS: list[tuple[str, str, bool]] = [
    (p.model, p.reasoning_effort, p.supports_developer_role) for p in LLM_PRESETS.values()
]
# Free-tier token-per-minute limits throttle back-to-back requests (each one
# carries ~1k tokens of system prompt), which would distort TTFT. Real calls
# have seconds between turns, so pace the requests similarly.
DEFAULT_PAUSE_SECS = 6.0
MAX_WORDS = 32  # the prompt says ~30; allow a little slack

GREETING = (
    "Hi, this is Alex from Brightfern Labs. I'm calling about Quillo, our AI meeting "
    "assistant. Do you spend a lot of time in video meetings?"
)
TRIAL_OFFER = "Would you like to try the 14-day free trial of the Pro plan?"


@dataclass
class Scenario:
    name: str
    turns: list[tuple[str, str]]  # (role, content) after the greeting
    expect: str = ""  # "", "not_sure", "question", "goodbye"
    with_greeting: bool = True


SCENARIOS: list[Scenario] = [
    Scenario("greeting", [], with_greeting=False),
    Scenario("pitch", [("user", "Yes, a lot. Tell me everything about Quillo.")]),
    Scenario("value", [("user", "Yes. But how will it actually help me?")]),
    Scenario("discount", [("user", "Do you offer any discounts?")], expect="not_sure"),
    Scenario(
        "website",
        [("user", "Sure."), ("assistant", TRIAL_OFFER), ("user", "Yes. What's your website?")],
        expect="not_sure",
    ),
    # Live failure: after "Yes" to the trial, gpt-oss said "just visit our website".
    Scenario("trial_yes", [("user", "Sure."), ("assistant", TRIAL_OFFER), ("user", "Yes.")]),
    Scenario("first_no", [("user", "No, I'm not interested.")], expect="question"),
    Scenario(
        "second_no",
        [
            ("user", "No, I'm not interested."),
            ("assistant", "I understand. How do you handle meeting notes today?"),
            ("user", "Someone on my team types them up."),
            (
                "assistant",
                "Quillo would free that person to focus on the conversation and sends the "
                "summary automatically. Would that help your team?",
            ),
            ("user", "No thanks, I'm still not interested."),
        ],
        expect="goodbye",
    ),
    Scenario("opt_out", [("user", "Please stop, I have to go.")], expect="goodbye"),
]

PROMISE_RE = re.compile(
    r"\bI(?:'ll| will)\s+(?:send|email|arrange|set (?:it|you|that) up|sign you up|schedule|book)",
    re.IGNORECASE,
)
NOT_SURE_RE = re.compile(
    r"not sure|don['’]t know|do not know|don['’]t have|not covered|aren['’]t covered|"
    r"isn['’]t covered|follow up|follow-up",
    re.IGNORECASE,
)
GOODBYE_RE = re.compile(r"\bbye\b|goodbye|have a (?:great|good|nice|wonderful) day|take care", re.I)
# Asserting a website exists ("on our website") is invented; saying you don't
# have the URL is honest, so a bare mention of "website" isn't flagged.
INVENTED_CLAIM_RE = re.compile(
    r"\d+\s*%|\bhours?\b|(?:visit|on|at|via|through|go to) our (?:website|site)|www\.|\.com\b",
    re.IGNORECASE,
)
BRIEF_NUMBERS = set(re.findall(r"\d+", DEMO_BRIEF))


@dataclass
class Result:
    model: str
    scenario: str
    reply: str = ""
    ttft_ms: float | None = None
    problems: list[str] = field(default_factory=list)
    error: str = ""


def check(scenario: Scenario, reply: str) -> list[str]:
    problems = []
    words = len(re.findall(r"[\w'’-]+", reply))
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s])
    if words > MAX_WORDS:
        problems.append(f"{words} words")
    if sentences > 2:
        problems.append(f"{sentences} sentences")
    if reply.count("?") > 1:
        problems.append("more than one question")
    if re.search(r"[—;]| – ", reply):
        problems.append("dash/semicolon chaining")
    if PROMISE_RE.search(reply):
        problems.append("promises to act itself")
    if INVENTED_CLAIM_RE.search(reply):
        problems.append("claim/website not in brief")
    extra = set(re.findall(r"\d+", reply)) - BRIEF_NUMBERS
    if extra:
        problems.append(f"numbers not in brief: {', '.join(sorted(extra))}")
    if re.search(r"thank(?:s| you) for calling", reply, re.IGNORECASE):
        problems.append('"thanks for calling" on an outbound call')
    if scenario.expect == "not_sure" and not NOT_SURE_RE.search(reply):
        problems.append("should say it's not sure / offer a follow-up")
    if scenario.expect == "question" and "?" not in reply:
        problems.append("should ask a needs question")
    if scenario.expect == "goodbye" and (not GOODBYE_RE.search(reply) or "?" in reply):
        problems.append("should say goodbye without another question")
    return problems


def messages_for(scenario: Scenario, developer_role: bool) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": build_system_prompt(DEMO_BRIEF)},
        {"role": "developer" if developer_role else "user", "content": GREETING_INSTRUCTION},
    ]
    if scenario.with_greeting:
        messages.append({"role": "assistant", "content": GREETING})
    messages += [{"role": role, "content": content} for role, content in scenario.turns]
    return messages


def run_one(
    client: groq.Groq, model: str, effort: str, developer_role: bool, scenario: Scenario
) -> Result:
    result = Result(model=model, scenario=scenario.name)
    started = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages_for(scenario, developer_role),  # type: ignore[arg-type]
            reasoning_effort=effort,  # type: ignore[arg-type]
            max_completion_tokens=1024,
            stream=True,
        )
        parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if result.ttft_ms is None:
                    result.ttft_ms = (time.perf_counter() - started) * 1000
                parts.append(delta)
        result.reply = "".join(parts).strip()
        result.problems = check(scenario, result.reply)
    except groq.APIError as error:
        status = getattr(error, "status_code", "")
        result.error = f"HTTP {status}: {str(error)[:160]}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2, help="runs per scenario per model")
    parser.add_argument(
        "--pause", type=float, default=DEFAULT_PAUSE_SECS, help="seconds between requests"
    )
    parser.add_argument("--out", type=Path, help="also save the full output as a markdown log")
    args = parser.parse_args()

    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line, flush=True)
        lines.append(line)
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Config error: {error}", file=sys.stderr)
        return 1
    client = groq.Groq(api_key=settings.groq_api_key)

    summary = []
    for model, effort, developer_role in MODELS:
        emit(f"\n=== {model} (reasoning_effort={effort}) ===")
        results = []
        for scenario in SCENARIOS:
            for _ in range(args.runs):
                results.append(run_one(client, model, effort, developer_role, scenario))
                time.sleep(args.pause)
        for r in results:
            if r.error:
                emit(f"  ERROR {r.scenario:<10} {r.error}")
                continue
            status = "PASS" if not r.problems else "FAIL"
            ttft = f"{r.ttft_ms:5.0f} ms" if r.ttft_ms else "   n/a"
            emit(f"  {status} {r.scenario:<10} {ttft}  {r.reply!r}")
            for problem in r.problems:
                emit(f"         - {problem}")
        ok = [r for r in results if not r.error]
        passed = sum(1 for r in ok if not r.problems)
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms]
        summary.append(
            (
                model,
                f"{passed}/{len(ok)}",
                f"{statistics.median(ttfts):.0f} ms" if ttfts else "n/a",
                sum(1 for r in results if r.error),
            )
        )

    emit("\n=== Summary ===")
    emit(f"  {'model':<22} {'passed':>8} {'median TTFT':>12} {'errors':>7}")
    for model, passed, ttft, errors in summary:
        emit(f"  {model:<22} {passed:>8} {ttft:>12} {errors:>7}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Persona eval, {datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"`uv run python scripts/eval_persona.py --runs {args.runs} --pause {args.pause}`\n\n"
        )
        args.out.write_text(header + "```\n" + "\n".join(lines).strip() + "\n```\n")
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
