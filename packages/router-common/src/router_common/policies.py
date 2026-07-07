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
    force_tier: str = "hard_dev"
    patterns: list[str] = Field(default_factory=list)


class EscalationConfig(BaseModel):
    max_escalations: int = 2
    retry_patterns: list[str] = Field(default_factory=list)
    failure_markers: list[str] = Field(default_factory=list)
    consecutive_tool_failures: int = 2


class SessionConfig(BaseModel):
    ttl_seconds: int = 28800
    redis_key_prefix: str = "router:session:"


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


def load_policies(path: str | Path) -> PoliciesConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PoliciesConfig.model_validate(raw)
