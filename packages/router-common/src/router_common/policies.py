"""Policy configuration models and loader for policies.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Tier(BaseModel):
    name: str
    model: str
    description: str = ""


class PathOverrides(BaseModel):
    force_tier: str = "high"
    patterns: list[str] = Field(default_factory=list)


class EscalationConfig(BaseModel):
    max_escalations: int = 2
    retry_patterns: list[str] = Field(default_factory=list)
    failure_markers: list[str] = Field(default_factory=list)
    consecutive_tool_failures: int = 2


class SessionConfig(BaseModel):
    ttl_seconds: int = 28800
    redis_key_prefix: str = "router:session:"


class SideChannelConfig(BaseModel):
    # A request whose system prompt contains any of these (case-insensitive)
    # substrings is a client housekeeping call — title generation, conversation
    # summarization, etc. — not the user's actual task. Such requests are routed
    # to the cheapest tier and NEVER classify or pin the session, so the real
    # conversation is what determines the pinned tier. (Agents like OpenCode and
    # Claude Code fire these on a shared session id, which would otherwise pin
    # the whole session to the trivial-task tier.)
    system_patterns: list[str] = Field(default_factory=list)


class ModelPricing(BaseModel):
    in_: float = Field(alias="in")
    out: float

    model_config = {"populate_by_name": True}


class PoliciesConfig(BaseModel):
    version: int = 1
    default_model: str
    tiers: list[Tier]
    path_overrides: PathOverrides = Field(default_factory=PathOverrides)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    side_channels: SideChannelConfig = Field(default_factory=SideChannelConfig)
    pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "PoliciesConfig":
        if not self.tiers:
            raise ValueError("policies must define at least one tier")
        names = [t.name for t in self.tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tier names: {names}")
        if self.path_overrides.force_tier not in names:
            raise ValueError(
                f"path_overrides.force_tier {self.path_overrides.force_tier!r} is not a tier"
            )
        return self

    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]

    def tier_by_name(self, name: str) -> Tier | None:
        for t in self.tiers:
            if t.name == name:
                return t
        return None

    def tier_index(self, name: str) -> int:
        for i, t in enumerate(self.tiers):
            if t.name == name:
                return i
        raise KeyError(name)

    def default_tier(self) -> Tier:
        """Tier whose model is default_model; falls back to the middle tier."""
        for t in self.tiers:
            if t.model == self.default_model:
                return t
        return self.tiers[len(self.tiers) // 2]

    def cheapest_tier(self) -> Tier:
        """Cheapest tier — tiers are ordered cheapest -> most capable."""
        return self.tiers[0]

    def is_side_channel(self, system_text: str) -> bool:
        """True if a request's system prompt marks it as client housekeeping."""
        if not system_text:
            return False
        low = system_text.lower()
        return any(p.lower() in low for p in self.side_channels.system_patterns)


def load_policies(path: str | Path) -> PoliciesConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PoliciesConfig.model_validate(raw)
