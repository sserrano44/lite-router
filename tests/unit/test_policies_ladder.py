"""Exercises the shipped example policies.yaml (repo root), separate from the
frozen unit fixture — so the 4-tier provider-agnostic ladder is validated."""

from pathlib import Path

from router_common.policies import load_policies

from session_router import state_machine as sm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _root_policies():
    return load_policies(REPO_ROOT / "policies.yaml")


def test_ladder_shape():
    p = _root_policies()
    assert [t.name for t in p.tiers] == ["quick_lookup", "standard_dev", "high", "ultra-think"]
    assert [t.model for t in p.tiers] == [
        "claude-sonnet-5", "grok-4.5", "claude-opus-4-8", "claude-fable-5",
    ]
    assert p.default_model == "grok-4.5"
    assert p.default_tier().name == "standard_dev"
    assert p.path_overrides.force_tier == "high"
    assert p.escalation.max_escalations == 3


def _pin(p, tier_name):
    tier = p.tier_by_name(tier_name)
    return sm.build_pin_record(
        p, classify=sm.ClassifyResult(tier.name, tier.model, 1.0),
        path_override=False, first_message_hash="h", api_key_alias="a",
        msg_count=1, now=0.0,
    )


def test_escalation_walks_full_ladder():
    p = _root_policies()
    rec = _pin(p, "quick_lookup")
    assert rec["model"] == "claude-sonnet-5"

    models = [rec["model"]]
    for _ in range(5):  # more than max_escalations to prove the cap holds
        out = sm.escalate(rec, p)
        if out is None:
            break
        rec = out
        models.append(rec["model"])

    # 3 hops off quick_lookup reaches the top tier, then no further escalation.
    assert models == ["claude-sonnet-5", "grok-4.5", "claude-opus-4-8", "claude-fable-5"]
    assert rec["policy_name"] == "ultra-think"
    assert rec["escalations"] == 3


def test_cap_reached_before_top_when_starting_higher():
    p = _root_policies()
    rec = _pin(p, "quick_lookup")
    # max_escalations=3 from the cheapest tier just reaches ultra-think;
    # starting at standard_dev, 3 hops would overshoot, so the cap stops it.
    rec = _pin(p, "standard_dev")
    for _ in range(5):
        out = sm.escalate(rec, p)
        if out is None:
            break
        rec = out
    assert rec["escalations"] <= p.escalation.max_escalations
