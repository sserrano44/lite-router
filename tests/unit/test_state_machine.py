from hypothesis import given, strategies as st

from session_router import state_machine as sm


def _pin(policies, **kwargs):
    defaults = dict(
        classify=None, path_override=False, first_message_hash="fmh",
        api_key_alias="alias", msg_count=1, now=1000.0,
    )
    defaults.update(kwargs)
    return sm.build_pin_record(policies, **defaults)


class TestBuildPinRecord:
    def test_classified(self, policies):
        rec = _pin(policies, classify=sm.ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
        assert rec["state"] == sm.STATE_PINNED
        assert rec["policy_name"] == "quick_lookup"
        assert rec["model"] == "claude-haiku-4-5"
        assert rec["classified"] is True
        assert rec["confidence"] == 0.9
        assert rec["escalations"] == 0

    def test_path_override_beats_classifier(self, policies):
        rec = _pin(
            policies,
            classify=sm.ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9),
            path_override=True,
        )
        assert rec["policy_name"] == "hard_dev"
        assert rec["model"] == "claude-opus-4-8"
        assert rec["path_override"] is True
        assert rec["classified"] is False

    def test_no_classifier_defaults(self, policies):
        rec = _pin(policies)
        assert rec["model"] == policies.default_model
        assert rec["policy_name"] == "standard_dev"
        assert rec["classified"] is False
        assert rec["confidence"] == 0.0

    def test_unknown_policy_from_classifier_defaults(self, policies):
        rec = _pin(policies, classify=sm.ClassifyResult("bogus_tier", "whatever", 0.5))
        assert rec["model"] == policies.default_model
        assert rec["classified"] is False


class TestEscalate:
    def test_ladder(self, policies):
        rec = _pin(policies, classify=sm.ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
        rec = sm.escalate(rec, policies)
        assert rec["policy_name"] == "standard_dev" and rec["escalations"] == 1
        rec = sm.escalate(rec, policies)
        assert rec["policy_name"] == "hard_dev" and rec["escalations"] == 2
        assert sm.escalate(rec, policies) is None  # capped AND at top

    def test_cap_at_max_escalations(self, policies):
        rec = _pin(policies, classify=sm.ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
        rec["escalations"] = policies.escalation.max_escalations
        assert sm.escalate(rec, policies) is None

    def test_top_tier_cannot_escalate(self, policies):
        rec = _pin(policies, path_override=True)
        assert sm.escalate(rec, policies) is None

    def test_classifying_placeholder_cannot_escalate(self, policies):
        assert sm.escalate(sm.classifying_placeholder(policies), policies) is None

    def test_unknown_policy_reanchors_to_default(self, policies):
        rec = _pin(policies)
        rec["policy_name"] = "removed_tier"
        out = sm.escalate(rec, policies)
        assert out is not None and out["policy_name"] == "hard_dev"


@given(st.lists(st.booleans(), max_size=10),
       st.sampled_from(["quick_lookup", "standard_dev", "hard_dev"]))
def test_tier_monotonic_non_decreasing(escalation_attempts, start_tier):
    """Property: no sequence of operations can ever lower a session's tier."""
    from pathlib import Path

    from router_common.policies import load_policies

    repo_root = Path(__file__).resolve().parent.parent.parent
    policies = load_policies(repo_root / "policies.yaml")
    tier = policies.tier_by_name(start_tier)
    rec = sm.build_pin_record(
        policies, classify=sm.ClassifyResult(tier.name, tier.model, 1.0),
        path_override=False, first_message_hash="h", api_key_alias="a",
        msg_count=1, now=0.0,
    )
    last_idx = policies.tier_index(rec["policy_name"])
    for _ in escalation_attempts:
        out = sm.escalate(rec, policies)
        if out is not None:
            rec = out
        idx = policies.tier_index(rec["policy_name"])
        assert idx >= last_idx
        last_idx = idx
    assert rec["escalations"] <= policies.escalation.max_escalations


class TestSubagentTier:
    def test_one_below_parent(self, policies):
        parent = _pin(policies)  # standard_dev
        assert sm.subagent_tier(policies, parent).name == "quick_lookup"

    def test_floor_at_cheapest(self, policies):
        parent = _pin(policies, classify=sm.ClassifyResult("quick_lookup", "claude-haiku-4-5", 1.0))
        assert sm.subagent_tier(policies, parent).name == "quick_lookup"

    def test_path_override_parent_keeps_tier(self, policies):
        parent = _pin(policies, path_override=True)
        assert sm.subagent_tier(policies, parent).name == "hard_dev"

    def test_hard_dev_classified_parent_goes_down(self, policies):
        parent = _pin(policies, classify=sm.ClassifyResult("hard_dev", "claude-opus-4-8", 1.0))
        assert sm.subagent_tier(policies, parent).name == "standard_dev"
