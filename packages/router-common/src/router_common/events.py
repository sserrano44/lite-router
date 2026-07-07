"""Decision-event types shared between the hook and the logging pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    CLASSIFIED = "classified"
    PINNED = "pinned"
    ESCALATED = "escalated"
    OVERRIDE = "override"
    FALLBACK = "fallback"


@dataclass(slots=True)
class DecisionEvent:
    session_key: str
    event_type: EventType
    model: str
    policy_name: str | None = None
    confidence: float | None = None
    first_message_hash: str | None = None
    api_key_alias: str | None = None
    latency_ms: int | None = None
    shadow: bool = False
    agent_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    # Only set on `classified` events when raw-message capture is enabled.
    raw_first_message: str | None = None
    system_excerpt: str | None = None
