"""Session pin/escalate transitions. Pure — the unit-testable core.

Invariant: a session's tier is monotonic non-decreasing. The only
model-changing transitions are the initial pin and `escalate`, which moves
exactly one tier up. No function here can produce a downgrade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from router_common.policies import PoliciesConfig, Tier

from session_router.escalation import ScanState

STATE_CLASSIFYING = "classifying"
STATE_PINNED = "pinned"
RECORD_VERSION = 1


@dataclass(slots=True)
class ClassifyResult:
    policy_name: str
    model: str
    confidence: float
    latency_ms: int | None = None


def classifying_placeholder(policies: PoliciesConfig) -> dict:
    """Record claimed via SET NX while the winner runs the classifier."""
    return {
        "v": RECORD_VERSION,
        "state": STATE_CLASSIFYING,
        "model": policies.default_model,
        "policy_name": policies.default_tier().name,
    }


def build_pin_record(
    policies: PoliciesConfig,
    *,
    classify: ClassifyResult | None,
    path_override: bool,
    first_message_hash: str,
    api_key_alias: str,
    msg_count: int,
    now: float | None = None,
) -> dict:
    """Initial pin. Precedence: path_override > classifier result > default."""
    if path_override:
        tier = policies.tier_by_name(policies.path_overrides.force_tier)
        assert tier is not None  # validated at policy load
        policy_name, model, confidence, classified = tier.name, tier.model, 1.0, False
    elif classify is not None and policies.tier_by_name(classify.policy_name):
        policy_name = classify.policy_name
        model = policies.tier_by_name(classify.policy_name).model  # type: ignore[union-attr]
        confidence, classified = classify.confidence, True
    else:
        tier = policies.default_tier()
        policy_name, model, confidence, classified = tier.name, policies.default_model, 0.0, False

    return {
        "v": RECORD_VERSION,
        "state": STATE_PINNED,
        "model": model,
        "policy_name": policy_name,
        "classified": classified,
        "confidence": confidence,
        "path_override": path_override,
        "escalations": 0,
        "pinned_at": now if now is not None else time.time(),
        "first_message_hash": first_message_hash,
        "api_key_alias": api_key_alias,
        "scan": ScanState(msg_count=msg_count).to_dict(),
    }


def escalate(record: dict, policies: PoliciesConfig) -> dict | None:
    """One tier up, sticky. None when capped, at the top, or not yet pinned."""
    if record.get("state") != STATE_PINNED:
        return None
    if int(record.get("escalations", 0)) >= policies.escalation.max_escalations:
        return None
    try:
        idx = policies.tier_index(record["policy_name"])
    except KeyError:
        # Unknown tier (policies changed under us): re-anchor to default tier.
        idx = policies.tier_index(policies.default_tier().name)
    if idx >= len(policies.tiers) - 1:
        return None
    new_tier = policies.tiers[idx + 1]
    updated = dict(record)
    updated["policy_name"] = new_tier.name
    updated["model"] = new_tier.model
    updated["escalations"] = int(record.get("escalations", 0)) + 1
    return updated


def subagent_tier(policies: PoliciesConfig, parent_record: dict) -> Tier:
    """R1b: one tier below the parent's pin, floor = cheapest tier.

    Path-override (mission-critical) parents are the exception: their
    subagents inherit the forced tier — R12b's "never downgrade" outranks the
    subagent discount for contracts/custody/bridge work.
    """
    try:
        idx = policies.tier_index(parent_record["policy_name"])
    except KeyError:
        idx = policies.tier_index(policies.default_tier().name)
    if parent_record.get("path_override"):
        return policies.tiers[idx]
    return policies.tiers[max(0, idx - 1)]
