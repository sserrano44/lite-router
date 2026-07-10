"""Escalation signal detection. Pure, bounded to the last few messages.

Anthropic /v1/messages bodies carry the full history on every request, so the
scan is driven by a watermark (`msg_count`) stored in the session record:
each message index is regex-scanned at most once over the session's lifetime,
and the window never exceeds MAX_TAIL messages even after a long gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from router_common.policies import EscalationConfig

from session_router.session_key import ESCALATE_HEADER, flatten_content

MAX_TAIL = 10
RETRY_TEXT_LIMIT = 500
TOOL_RESULT_LIMIT = 2000


@dataclass(slots=True)
class ScanState:
    msg_count: int = 0
    consec_failures: int = 0
    escalated_at_msg_count: int = 0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ScanState":
        raw = raw or {}
        return cls(
            msg_count=int(raw.get("msg_count", 0)),
            consec_failures=int(raw.get("consec_failures", 0)),
            escalated_at_msg_count=int(raw.get("escalated_at_msg_count", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "msg_count": self.msg_count,
            "consec_failures": self.consec_failures,
            "escalated_at_msg_count": self.escalated_at_msg_count,
        }


@dataclass(slots=True)
class EscalationSignal:
    reason: str  # "header" | "retry_text" | "tool_failures"
    detail: dict = field(default_factory=dict)


@lru_cache(maxsize=8)
def _compiled(patterns: tuple[str, ...], flags: int = 0) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, flags) for p in patterns)


def _matches_retry(text: str, patterns: tuple[re.Pattern, ...]) -> str | None:
    snippet = text[:RETRY_TEXT_LIMIT].strip()
    for pat in patterns:
        if pat.pattern.startswith("^"):
            if pat.match(snippet):
                return pat.pattern
        elif pat.search(snippet):
            return pat.pattern
    return None


def _tool_result_blocks(msg: dict) -> list[dict]:
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [
        b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


def _block_failed(block: dict, failure_re: re.Pattern) -> bool:
    if block.get("is_error") is True:
        return True
    text = flatten_content(block.get("content"), TOOL_RESULT_LIMIT)
    return bool(text and failure_re.search(text))


def _tool_result_failures(msg: dict, failure_re: re.Pattern) -> list[bool] | None:
    """Per-tool-result failure flags for a tool turn, or None if not one.

    Anthropic shape: a role=="user" message carrying tool_result blocks (each
    may set is_error or contain a failure marker). OpenAI shape: a role=="tool"
    message whose string content matches a failure marker (no is_error flag).
    Returning None means "not a tool turn" so the failure streak is untouched.
    """
    role = msg.get("role")
    if role == "user":
        blocks = _tool_result_blocks(msg)
        if blocks:
            return [_block_failed(b, failure_re) for b in blocks]
        return None
    if role == "tool":
        text = flatten_content(msg.get("content"), TOOL_RESULT_LIMIT)
        return [bool(text and failure_re.search(text))]
    return None


def detect(
    messages: list,
    headers: dict[str, str],
    scan: ScanState,
    cfg: EscalationConfig,
) -> tuple[EscalationSignal | None, ScanState]:
    """Return (signal or None, updated scan state).

    The caller is responsible for persisting the returned state, and — when it
    applies an escalation — for calling `consume(state, n)` so the same
    evidence never fires twice.
    """
    if headers.get(ESCALATE_HEADER, "").strip().lower() == "true":
        n = len(messages) if isinstance(messages, list) else scan.msg_count
        return EscalationSignal("header"), ScanState(
            msg_count=max(n, scan.msg_count),
            consec_failures=0,
            escalated_at_msg_count=max(n, scan.msg_count),
        )

    if not isinstance(messages, list):
        return None, scan

    n = len(messages)
    if n < scan.msg_count:
        # History shrank (compaction / edited conversation): reset watermark
        # to a bounded tail rescan and drop stale counters.
        scan = ScanState(msg_count=max(0, n - MAX_TAIL), consec_failures=0,
                         escalated_at_msg_count=0)
    start = max(scan.msg_count, n - MAX_TAIL)

    retry_res = _compiled(tuple(cfg.retry_patterns), re.IGNORECASE)
    failure_re_list = _compiled((("|".join(cfg.failure_markers)) or r"(?!x)x",))
    failure_re = failure_re_list[0]

    consec = scan.consec_failures
    signal: EscalationSignal | None = None

    for i in range(start, n):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        failures = _tool_result_failures(msg, failure_re)
        if failures is not None:
            for failed in failures:
                consec = consec + 1 if failed else 0
            if not signal and consec >= cfg.consecutive_tool_failures:
                signal = EscalationSignal("tool_failures", {"consecutive": consec})
        # Retry text: only the final message, only when it is a plain (non-tool)
        # user message past the last escalation point.
        if (
            i == n - 1
            and failures is None
            and msg.get("role") == "user"
            and i >= scan.escalated_at_msg_count
        ):
            text = flatten_content(msg.get("content"), RETRY_TEXT_LIMIT)
            hit = _matches_retry(text, retry_res)
            if hit and not signal:
                signal = EscalationSignal("retry_text", {"pattern": hit})

    return signal, ScanState(
        msg_count=n,
        consec_failures=consec,
        escalated_at_msg_count=scan.escalated_at_msg_count,
    )


def consume(state: ScanState, message_count: int) -> ScanState:
    """Mark an applied escalation: evidence up to `message_count` is spent."""
    return ScanState(
        msg_count=max(state.msg_count, message_count),
        consec_failures=0,
        escalated_at_msg_count=max(state.escalated_at_msg_count, message_count),
    )
