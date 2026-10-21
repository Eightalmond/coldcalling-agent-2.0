"""Usage guardrails that keep a public demo inside the free tiers.

- Rate limits: calls per visitor per hour, calls per day for everyone, and
  brief generations per visitor per hour (each one is a Groq request).
- Call registry: a call must be created (POST /api/calls, which validates the
  brief and checks limits) before a browser may open a WebRTC connection,
  using a one-time call id. This also caps concurrent calls.

Everything is in memory: limits reset when the server restarts. That's fine
for a single-instance demo; a multi-instance deployment would need Redis.
"""

import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# Per-call cap (enforced in the bot by CallTimer) and request limits.
CALL_MAX_SECS = 5 * 60
CALLS_PER_VISITOR_PER_HOUR = 3
CALLS_PER_DAY = 30
BRIEFS_PER_VISITOR_PER_HOUR = 5
MAX_CONCURRENT_CALLS = 2
# A created call must be connected within this time, or its id expires.
CALL_CONNECT_TIMEOUT_SECS = 60

HOUR = 60 * 60
DAY = 24 * HOUR

Clock = Callable[[], float]


class SlidingWindowLimiter:
    """At most `limit` events per key in any `window_secs` window."""

    def __init__(self, limit: int, window_secs: float, clock: Clock = time.monotonic) -> None:
        self.limit = limit
        self.window_secs = window_secs
        self._clock = clock
        self._events: dict[str, deque[float]] = {}

    def _prune(self, key: str) -> deque[float]:
        events = self._events.setdefault(key, deque())
        cutoff = self._clock() - self.window_secs
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may act again (0 if it may act now)."""
        events = self._prune(key)
        if len(events) < self.limit:
            return 0.0
        return max(0.0, events[0] + self.window_secs - self._clock())

    def record(self, key: str) -> None:
        self._prune(key).append(self._clock())


class LimitExceeded(Exception):
    """A guardrail refused the request; `retry_after` is in seconds."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class Call:
    call_id: str
    brief: str
    created_at: float
    connected: bool = False


@dataclass
class CallRegistry:
    """Pending and active calls, with the call limits checked on create()."""

    clock: Clock = time.monotonic
    max_concurrent: int = MAX_CONCURRENT_CALLS
    connect_timeout_secs: float = CALL_CONNECT_TIMEOUT_SECS
    per_visitor: SlidingWindowLimiter = field(init=False)
    per_day: SlidingWindowLimiter = field(init=False)
    _calls: dict[str, Call] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.per_visitor = SlidingWindowLimiter(CALLS_PER_VISITOR_PER_HOUR, HOUR, self.clock)
        self.per_day = SlidingWindowLimiter(CALLS_PER_DAY, DAY, self.clock)

    def _drop_expired(self) -> None:
        cutoff = self.clock() - self.connect_timeout_secs
        for call_id in [c.call_id for c in self._calls.values() if not c.connected]:
            if self._calls[call_id].created_at <= cutoff:
                del self._calls[call_id]

    @property
    def open_calls(self) -> int:
        self._drop_expired()
        return len(self._calls)

    def create(self, brief: str, visitor: str) -> Call:
        """Check every call limit, then register a new pending call."""
        if self.open_calls >= self.max_concurrent:
            raise LimitExceeded("All demo lines are busy right now. Try again in a minute.", 30)
        wait = self.per_visitor.retry_after(visitor)
        if wait:
            raise LimitExceeded(
                f"You've reached {CALLS_PER_VISITOR_PER_HOUR} calls this hour. Try again later.",
                wait,
            )
        wait = self.per_day.retry_after("all")
        if wait:
            raise LimitExceeded("The demo has reached today's call limit. Try again tomorrow.", wait)

        self.per_visitor.record(visitor)
        self.per_day.record("all")
        call = Call(call_id=secrets.token_urlsafe(16), brief=brief, created_at=self.clock())
        self._calls[call.call_id] = call
        return call

    def claim(self, call_id: str) -> Call | None:
        """Mark a pending call connected. Returns None if unknown, expired or used."""
        self._drop_expired()
        call = self._calls.get(call_id)
        if call is None or call.connected:
            return None
        call.connected = True
        return call

    def finish(self, call_id: str) -> None:
        self._calls.pop(call_id, None)
